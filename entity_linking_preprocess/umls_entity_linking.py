import sys

sys.path.append("/nfs")
sys.path.append("/nfs/general")
sys.path.append("/nfs/long_tail")

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

import pickle 
import numpy as np 
import pandas as pd
from glob import glob 
from tqdm import tqdm, trange
from multiprocessing import Pool

import torch
from transformers import AutoTokenizer, AutoModel

from utils.index import Indexer, build_entity_embedding_index, generate_umls_embedding

from utils.my_utils import check_file_exists

# from long_tail.datasets.pubtator.pubtator_processor import PubtatorProcessor, MultiPubtatorProcessor
from data.pubtator.pubtator_processor import MultiPubtatorProcessor
from dsets import PubTatorDataset
from dsets import UMLSDataset



def sapbert_entity_linking(query_list, topk=1, batch_size=1024):
    # each query should be a mention
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
        # 我们只要第一个token的embedding，因为sapbert就是用的第一个token的embedding去代表的这个entity。
        batch_query_embedding = model(**inputs)[0][:, 0]
        # cls_rep = model(**toks_cuda)[0][:,0,:]
        batch_query_embedding = torch.nn.functional.normalize(batch_query_embedding, dim=1) # 对向量进行归一化
        # 用于创建一个新的张量（Tensor），该张量与原始张量共享相同的数据，但不再与计算图关联。这个方法的主要目的是分离张量，使得你可以在不影响梯度计算的情况下对其进行操作。
        query_embedding_list.append(batch_query_embedding.cpu().detach().numpy())

    query_embeddings = np.concatenate(query_embedding_list, axis=0)

    indexer = Indexer(embedding_size, metric="inner_product")
    index_path = "/nfs/long_tail/data/umls/metathesaurus/index"
    print(f"Loading index from {index_path} ...")
    indexer.deserialize_from(index_path)

    print("Entity Linking ...")
    results = indexer.search_knn(
        query_vectors=query_embeddings,
        top_docs=topk, 
        index_batch_size=batch_size
    )
    """
    [
        [('168113', 0.69278115), ('352272', 0.61756897), ('217145', 0.6094995)], 
        [('65713', 0.68059087), ('340646', 0.67876977), ('364605', 0.67388695)], 
        [('127568', 0.6484056), ('127541', 0.63720894), ('191111', 0.63089496)], 
        [('173760', 0.99999994), ('173765', 0.9921722), ('95654', 0.89093053)], 
        [('433799', 0.86873823), ('92394', 0.8502206), ('293822', 0.8179185)], 
        [('348926', 0.84749866), ('99038', 0.8402275), ('350606', 0.8162697)], 
        [('191388', 0.9277023), ('16027', 0.91782147), ('16028', 0.90698355)], 
        [('91902', 0.9999999), ('92501', 0.9762863), ('181433', 0.899636)], 
        [('153945', 0.9005642), ('109368', 0.883803), ('263214', 0.8809234)], 
        [('160324', 1.0), ('48288', 0.9796798), ('160312', 0.84929454)]
    ]
    """

    outputs = []
    for topk_str_indices, topk_score_array in results:
        outputs.append([(int(idx), score) for idx, score in zip(topk_str_indices, topk_score_array)])

    return outputs

def __generate_midx2mention_from_mentions(mentions):
    midx2mention = {midx : mention for midx, mention in enumerate(mentions)}
    return midx2mention

def __generate_id2midx_from_entity_linking_outputs_with_indices_(id2midxs, outputs, indices, threshold=0.85):
    if len(outputs) != len(indices):
        raise ValueError("In method \"generate_id2midx_from_entity_linking_outputs_with_indices(outputs, indices, threshold)\", \"outputs\" and \"indices\" must have the same length")
    
    print(f"Filtering the outputs of entity linking and remove the items with score less than threshold: {threshold} ...")
    midx_total_set = set()
    nid = 0
    for midx, tuple_list in zip(indices, outputs):
        for tuple in tuple_list:
            id = tuple[0]
            score = tuple[1]
            if score > threshold:
                if id not in id2midxs.keys():
                    id2midxs[id] = set()
                id2midxs[id].add(midx)
    for id, midxs in id2midxs.items():
        midx_total_set.update(midxs)
        nid += 1
    nmidx = len(midx_total_set)
    print(f"Statistics of id2midxs dictionary: The number of id = {nid}, the number of mention index = {nmidx}")
    
    return id2midxs


# batch_size=(10 ** 6)
def get_id2mentions_from_sapbert_entity_linking_batch_queries(mention2PMIDs, file_name, batch_size=10**6):
    
    mention_list = list(mention2PMIDs.keys())
    midxs = [index for index in range(len(mention_list))]
    
    # id 是entity的id，midx是mention的index，因为第一个entity可能映射到多个mention的index，所以是id2midxs
    id2midxs = dict()
    
    print(f"Total num of mention list is: {len(mention_list)}. The batch size is {batch_size}. There'll be {((len(mention_list)-1) // batch_size + 1) + 1} rounds.")
    
    save_data_path = "/nfs/long_tail/entity_linking_preprocess"
    if not check_file_exists(os.path.join("/nfs/long_tail/entity_linking_preprocess", "id2mentions.pickle")):
        
        # Step 1: calculate id2midxs
        for i in trange((len(mention_list)-1) // batch_size + 1):
            print(f"ROUND {i} ...")
            batch_mention_list = mention_list[i*batch_size: (i+1)*batch_size]
            batch_midxs = midxs[i*batch_size: (i+1)*batch_size]
            
            batch_outputs = sapbert_entity_linking(query_list=batch_mention_list, topk=1)
            id2midxs = __generate_id2midx_from_entity_linking_outputs_with_indices_(id2midxs=id2midxs, outputs=batch_outputs, indices=batch_midxs)
            # id2midxs = generate_id2midx_from_entity_linking_outputs(outputs=batch_outputs)
            
        pickle.dump(id2midxs, open(os.path.join(save_data_path, f"{file_name}_id2midxs.pickle"), mode="wb"))
        
        # Step 2: calculate midx2mention
        midx2mention = __generate_midx2mention_from_mentions(mentions=mention_list)
        pickle.dump(midx2mention, open(os.path.join(save_data_path, f"{file_name}_midx2mention.pickle"), mode="wb"))
        
        # step 3: calculate id2mentions
        id2mentions = dict()
        for id, midxs in id2midxs.items():
            for mid in midxs:
                mention = midx2mention[mid]
                if id not in id2mentions.keys():
                    id2mentions[id] = set()
                id2mentions[id].add(mention)
            
        pickle.dump(id2mentions, open(os.path.join(save_data_path, f"{file_name}_id2mentions.pickle"), mode="wb"))
    
    else:
        id2mentions = pickle.load(open(os.path.join(save_data_path, f"{file_name}_id2mentions.pickle"), mode="rb"))
    
    return id2mentions

def get_id2PMIDs(id2mentions, mention2PMIDs):
    id2PMIDs = dict()
    for id, mentions in id2mentions.items():
        PMID_set = set()
        for mention in mentions:
            PMIDs = mention2PMIDs[mention]
            PMID_set.update(PMIDs)
        
        if id not in id2PMIDs.keys():
            id2PMIDs[id] = set()
        id2PMIDs[id].update(PMID_set)
    
    save_data_path = "/nfs/long_tail/entity_linking_preprocess"
    pickle.dump(id2PMIDs, open(os.path.join(save_data_path, f"{file_name}_id2PMIDs.pickle"), mode="wb"))


def get_triple2coexistPMIDs(triples, ent2id, id2PMIDs):
    save_data_path = "/nfs/long_tail/entity_linking_preprocess"
    n_triples = len(triples)
    triple2coexistPMIDs = dict()
    for triple in tqdm(triples, desc="Scanning all the triples ...", total=n_triples):
        s_eid = triple[0]
        # r_id = triple[1]
        o_eid = triple[2]
        
        # from pdb import set_trace; set_trace()
        
        sid = ent2id[s_eid]
        oid = ent2id[o_eid]
        
        if sid in id2PMIDs.keys() and len(id2PMIDs[sid]) > 0 and oid in id2PMIDs.keys() and len(id2PMIDs[oid]) > 0:
            s_PMIDs = id2PMIDs[sid]
            o_PMIDs = id2PMIDs[oid]
            # print(s_PMIDs)
            # print(o_PMIDs)
            # 如果没有重合的, coexist_PMIDs就是一个空的set()
            coexist_PMIDs = s_PMIDs.intersection(o_PMIDs)
            triple2coexistPMIDs[triple] = coexist_PMIDs
        else:
            # 至少有一s或者o根本就没法映射为知识图谱的某个节点，这个triple根本就是异常的，丢弃掉
            # len(id2PMIDs[sid]) == 0 或 len(id2PMIDs[oid]) == 0，也是有一s或者o根本就没法映射为知识图谱的某个节点，triple算异常
            # triple2coexistPMIDs[triple] = []
            pass
    
    pickle.dump(triple2coexistPMIDs, open(os.path.join(save_data_path, f"{file_name}_triple2coexistPMIDs.pickle"), mode="wb"))
    
    # statistics_id2mentions(file_name)
    # statistics_id2PMIDs(file_name)
    # statistics_triple2coexistPMIDs(file_name)


if __name__ == "__main__":
    
    pubtator_dataset = PubTatorDataset()
    # snomedCT_dataset = SnomedCTDataset()
    umls_dataset = UMLSDataset()
    mention2PMIDs:dict = pubtator_dataset.load_mention2PMIDs()
    file_name = pubtator_dataset.file_name
    
    ###############################################
    #      Entity Linking from Pubmed to umsl     #
    ###############################################
    
    print("###############################################")
    print("#      Entity Linking from Pubmed to umsl     #")
    print("###############################################")
    
    # step 1: Generate embeddings of UMLS entities
    """
    print("step 1: Generate embeddings of UMLS entities")
    generate_umls_embedding()
    """
    
    # step 2: Build index based on embeddings from UMLS entities
    """
    print("step 2: Build index based on embeddings from UMLS entities")
    build_entity_embedding_index()
    """
    
    
    # Step 3: Extarct the "triples" from usml
    """
    print("Step 3: Extarct the \"triples\" from usml")
    triples = umls_dataset.load_triples()
    """
    
    # Step 4: Extract the entity2id, id2entityname from usml
    """
    print("# Step 4: Extract the entity2id, id2entityname from usml.") 
    ent2id = umls_dataset.load_entity2id()
    id2entityname = umls_dataset.load_id2entityname()
    """
    
    # Step 5: Entity Linking via sapbert, generate id2midxs, midx2mention, id2mentions, id means the entity id
    """
    print("Step 5: Entity Linking via sapbert, generate id2mentions.")
    id2mentions = get_id2mentions_from_sapbert_entity_linking_batch_queries(mention2PMIDs, file_name)
    """
    
    # Step 6: get dictionary of \"entity id to PMIDs\"", i.e., id2PMIDs
    """
    print("Step 6: get dictionary of \"entity id to PMIDs\", i.e., id2PMIDs.")
    id2PMIDs = get_id2PMIDs(id2mentions=id2mentions, mention2PMIDs=mention2PMIDs)
    """
    
    # Step 7: Retrieve all the triples, for each triple (s, r, o), find the coexist PMIDs for s and o.
    
    print("Step 7: Generate triple2coexistPMIDs ----- Retrieve all the triples, for each triple (s, r, o), find the coexist PMIDs for s and o. ")
    triples = umls_dataset.load_triples()
    ent2id = umls_dataset.load_entity2id()
    save_data_path = "/nfs/long_tail/entity_linking_preprocess"
    id2PMIDs = pickle.load(open(os.path.join(save_data_path, f"{file_name}_id2PMIDs.pickle"), mode="rb"))
    triple2coexistPMIDs = get_triple2coexistPMIDs(triples=triples, ent2id=ent2id, id2PMIDs=id2PMIDs)
    