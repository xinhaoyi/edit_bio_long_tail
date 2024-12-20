# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# jy: copy from FiD code 

import os
import logging
import pickle
from typing import List, Tuple

# import faiss
import numpy as np
import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger()


class Indexer(object):

    def __init__(self, vector_sz, metric="inner_product", n_subquantizers=0, n_bits=8):
        import faiss
        FAISSINDEX_DICT = {
            "inner_product": faiss.IndexFlatIP,
            "l2": faiss.IndexFlatL2,
        }
        if n_subquantizers > 0:
            self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = FAISSINDEX_DICT[metric](vector_sz)
        self.index_id_to_db_id = np.empty((0), dtype=np.int64)

    def index_data(self, ids, embeddings):

        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float32')
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)
        
        logger.info(f'Total data indexed {len(self.index_id_to_db_id)}')

    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size=1024) -> List[Tuple[List[object], List[float]]]:
        query_vectors = query_vectors.astype('float32')
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        for k in tqdm(range(nbatch)):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def serialize(self, dir_path):
        import faiss
        # index_file = dir_path / 'index.faiss'
        # meta_file = dir_path / 'index_meta.dpr'
        index_file = os.path.join(dir_path, "index.faiss")
        meta_file = os.path.join(dir_path, "index_meta.faiss")
        logger.info(f'Serializing index to {index_file}, meta data to {meta_file}')

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        import faiss
        # index_file = dir_path / 'index.faiss'
        # meta_file = dir_path / 'index_meta.dpr'
        index_file = os.path.join(dir_path, "index.faiss")
        meta_file = os.path.join(dir_path, "index_meta.faiss")
        logger.info(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)
        logger.info('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        new_ids = np.array(db_ids, dtype=np.int64)
        self.index_id_to_db_id = np.concatenate((self.index_id_to_db_id, new_ids), axis=0)

def generate_umls_embedding():
    save_file = "/nfs/long_tail/data/umls/metathesaurus/process/sapbert_all_entity_embeddings.npy"
    print("loading id2entity name ... ")
    id2entityname = pickle.load(open("/nfs/long_tail/data/umls/metathesaurus/process/id2entityname.pickle", mode="rb"))
    entity_list = [id2entityname[i] for i in range(len(id2entityname))]

    print("Loading SpaBERT tokenizer and model .... ")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)

    entity_embedding_list = []
    max_entity_text_length = 32 
    batch_size = 128 
    for i in trange((len(entity_list)-1)//batch_size + 1):
        batch_entity_list = entity_list[i*batch_size: (i+1)*batch_size]
        inputs = tokenizer(batch_entity_list, padding=True, truncation=True, max_length=max_entity_text_length, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        batch_entity_embeddings = model(**inputs)[0][:, 0]
        batch_entity_embeddings = torch.nn.functional.normalize(batch_entity_embeddings, dim=1) # 对向量进行归一化
        entity_embedding_list.append(batch_entity_embeddings.cpu().detach().numpy())

    all_entity_embedding = np.concatenate(entity_embedding_list, axis=0)

    print(f"saving entity embeddings to {save_file} ...")
    np.save(save_file, all_entity_embedding)

# examples 

def build_entity_embedding_index():

    embedding_size = 768 
    indexer = Indexer(embedding_size)

    index_path = "/nfs/long_tail/data/umls/metathesaurus/index"
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    all_entity_embeddings = np.load("/nfs/long_tail/data/umls/metathesaurus/process/sapbert_all_entity_embeddings.npy")
    all_entity_ids = list(range(len(all_entity_embeddings)))
    buffer_size = 50000
    for i in trange((len(all_entity_ids)-1) // buffer_size + 1):
        batch_entity_ids = all_entity_ids[i*buffer_size: (i+1)*buffer_size]
        batch_entity_embeddings = all_entity_embeddings[i*buffer_size: (i+1)*buffer_size]
        indexer.index_data(batch_entity_ids, batch_entity_embeddings)
    
    indexer.serialize(index_path)
    print(f"Successfully save index to {index_path}!")


def sapbert_entity_linking(query_list, topk=1, batch_size=1024):
    
    embedding_size = 768 

    print("Loading SpaBERT tokenizer and model .... ")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)

    print("Generating query embeddings ...")
    query_batch_size = 128 
    max_entity_text_length = 32 
    query_embedding_list = []
    for i in trange((len(query_list)-1) // query_batch_size + 1):
        inputs = tokenizer(
            query_list[i*query_batch_size: (i+1)*query_batch_size],
            padding=True, 
            truncation=True, 
            max_length=max_entity_text_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        batch_query_embedding = model(**inputs)[0][:, 0]
        query_embedding_list.append(batch_query_embedding.cpu().detach().numpy())
    query_embeddings = np.concatenate(query_embedding_list, axis=0)

    indexer = Indexer(embedding_size)
    index_path = "/nfs/long_tail/datasets/usml/index"
    print(f"Loading index from {index_path} ...")
    indexer.deserialize_from(index_path)

    print("Entity Linking ...")
    results = indexer.search_knn(
        query_vectors=query_embeddings,
        top_docs=topk, 
        index_batch_size=batch_size
    )

    outputs = []
    for topk_str_indices, topk_score_array in results:
        outputs.append([(idx, score) for idx, score in zip(topk_str_indices, topk_score_array)])

    return outputs