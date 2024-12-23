"""
Extract the dictionary “mention2PMIDs” from pubtator. 
    mention2PMIDs: mention: str -> a set of PMID (each PMID represents a paper id)
Extarct the "triples" from usml > SnomedCT
    SNOMED CT is a structured clinical vocabulary for use in an electronic health record. 
    It is the most comprehensive and precise clinical health terminology product in the world.
Extract the entity2id, relation2id, id2entityname, id2relationname from usml > SnomedCT
After entity linking, we'll have mention2id, id2mentions.
Based on id2mentions and mention2PMIDs, we can generate id2PMIDs
Retrieve all the triples, for each triple (s, r, o), find the coexist PMIDs for s and o.
"""
import sys

from dsets.pubtator_dataset import PubTatorDataset
from dsets.knowledge_graph_dataset import SnomedCTDataset
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
from utils.index import Indexer

from utils.my_utils import check_file_exists

# from long_tail.datasets.pubtator.pubtator_processor import PubtatorProcessor, MultiPubtatorProcessor
from data.pubtator.pubtator_processor import MultiPubtatorProcessor
from data.pubtator.extract_PubMed_abstract_data import extarct_PubMed_data_via_PMID
from data.umls.sapbert_entity_linking_preprocessing_snomedCT import *




def sapbert_entity_linking(query_list, topk=1, batch_size=1024):
    # each query should be a mention
    embedding_size = 768 

    # print("Loading SpaBERT tokenizer and model .... ")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)

    # print("Generating query embeddings ...")
    query_batch_size = 128 
    max_entity_text_length = 32 
    query_embedding_list = []
    for i in range((len(query_list)-1) // query_batch_size + 1):
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
    index_path = "/nfs/long_tail/data/umls/SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z/process/index"
    # print(f"Loading index from {index_path} ...")
    indexer.deserialize_from(index_path)

    # print("Entity Linking ...")
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

# batch_size=(10 ** 6)
def get_id2mentions_from_sapbert_entity_linking_batch_queries(mention2PMIDs, file_name, batch_size=10**6):
    
    mention_list = list(mention2PMIDs.keys())
    midxs = [index for index in range(len(mention_list))]
    
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
            generate_id2midx_from_entity_linking_outputs_with_indices_(id2midxs=id2midxs, outputs=batch_outputs, indices=batch_midxs)
            # id2midxs = generate_id2midx_from_entity_linking_outputs(outputs=batch_outputs)
            
        pickle.dump(id2midxs, open(os.path.join(save_data_path, f"{file_name}_id2midxs.pickle"), mode="wb"))
        
        # Step 2: calculate midx2mention
        midx2mention = generate_midx2mention_from_mentions(mentions=mention_list)
        pickle.dump(midx2mention, open(os.path.join(save_data_path, f"{file_name}_midx2mention.pickle"), mode="wb"))
        
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
    
    
def generate_midx2mention_from_mentions(mentions):
    midx2mention = {midx : mention for midx, mention in enumerate(mentions)}
    return midx2mention

def generate_id2midx_from_entity_linking_outputs_with_indices_(id2midxs, outputs, indices, threshold=0.85):
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

# unused method
def generate_id2midx_from_entity_linking_outputs(outputs, threshold=0.8):
    print(f"Filtering the outputs of entity linking and remove the items with score less than threshold: {threshold} ...")
    id2midxs = dict()
    for midx, tuple_list in enumerate(outputs):
        for tuple in tuple_list:
            id = int(tuple[0])
            score = tuple[1]
            if score > threshold:
                if id not in id2midxs.keys():
                    id2midxs[id] = set()
                id2midxs[id].add(midx)
    
    midx_total_set = set()
    nid = 0
    for id, midxs in id2midxs.items():
        midx_total_set.update(midxs)
        nid += 1
    nmidx = len(midx_total_set)
    print(f"Statistics of id2midxs dictionary: The number of id = {nid}, the number of mention index = {nmidx}")
    
    return id2midxs



def main():
    file_name_list = ['cellline2pubtatorcentral', 'chemical2pubtatorcentral', 'disease2pubtatorcentral', 'gene2pubtatorcentral', 'mutation2pubtatorcentral', 'species2pubtatorcentral']
    
    # Step 1: Extract the dictionary “mention2PMIDs” from pubtator.
    #===========================================#
    #              single processor             #
    #===========================================#
    # print("Step 1: Extract the dictionary “mention2PMIDs” from pubtator.")
    # pub_processor = PubtatorProcessor('mutation2pubtatorcentral')
    # mention2PMIDs_name = f"{pub_processor.file_name}_mention2PMIDs.pickle"
    # mention2PMIDs_path = os.path.join(pub_processor.data_process_path, mention2PMIDs_name)
    # print(f"Loading mention2PMIDs from {mention2PMIDs_path}")
    # mention2PMIDs: dict = pub_processor.load_mention2PMIDs_from_saved_file()
    # file_name = pub_processor.file_name
    
    
    #===========================================#
    #              multi processor             #
    #===========================================#
    pubtator_dataset = PubTatorDataset()
    snomedCT_dataset = SnomedCTDataset()
    mention2PMIDs:dict = pubtator_dataset.load_mention2PMIDs()
    file_name = pubtator_dataset.file_name
    
    
    # Step 2: Extarct the "triples" from usml > SnomedCT
    print("Step 2: Extarct the \"triples\" from usml > SnomedCT.")
    save_snomedCT_data_path = "/nfs/long_tail/my_datasets/usml/SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z/process"
    triples = snomedCT_dataset.load_triples()
    
    # Step 3: Extract the entity2id, id2entityname from usml > SnomedCT
    print("# Step 3: Extract the entity2id, id2entityname from usml > SnomedCT.") 
    ent2id = snomedCT_dataset.load_entity2id()
    id2entityname = snomedCT_dataset.load_id2entityname()
    
    # Step 4: Entity Linking via sapbert, generate id2mentions
    print("Step 4: Entity Linking via sapbert, generate id2mentions.")
    
    id2mentions = get_id2mentions_from_sapbert_entity_linking_batch_queries(mention2PMIDs, file_name)
    
    # Step 5: get id2PMIDs
    print("Step 5: get id2PMIDs.")
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
    
    # Step 6: Retrieve all the triples, for each triple (s, r, o), find the coexist PMIDs for s and o.
    n_triples = len(triples)
    print(f"Step 6: Retrieve all the triples, for each triple (s, r, o) in total {n_triples} triples, find the coexist PMIDs for s and o.")
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
    

def temp_method_convert_str2int_wihin_dict():
    
    file_name_list = ['cellline2pubtatorcentral', 'chemical2pubtatorcentral', 'disease2pubtatorcentral', 'gene2pubtatorcentral', 'mutation2pubtatorcentral', 'species2pubtatorcentral']
    # pub_processor = PubtatorProcessor('mutation2pubtatorcentral')
    multi_pub_processor = MultiPubtatorProcessor(file_name_list)
    file_name = multi_pub_processor.merged_file_name
    
    save_data_path = "/nfs/long_tail/entity_linking_preprocess"
    # ids_in_id2mentions_int_format = [int(id) for id in id2mentions.keys()]
    id2midxs_path = os.path.join(save_data_path, f"{file_name}_id2midxs.pickle")
    id2mentions_path = os.path.join(save_data_path, f"{file_name}_id2mentions.pickle")
    id2PMIDs_path = os.path.join(save_data_path, f"{file_name}_id2PMIDs.pickle")
    

    # id2midxs这里面id是字符串，midx是数字
    id2midxs = pickle.load(open(id2midxs_path, mode="rb"))
    id2mentions = pickle.load(open(id2mentions_path, mode="rb"))
    id2PMIDs = pickle.load(open(id2PMIDs_path, mode="rb"))
    
    id2midxs_edit = {}
    # {int(key): value for key, value in id2midxs.items()}
    for id, midxs in tqdm(id2midxs.items(), desc=f"processing the id2midxs", total=len(id2midxs)):
        id2midxs_edit[int(id)] = midxs
    pickle.dump(id2midxs_edit, open(id2midxs_path, mode="wb"))
    
    
    id2mentions_edit = {}
    for id, mentions in tqdm(id2mentions.items(), desc=f"processing the id2mentions", total=len(id2mentions)):
        id2mentions_edit[int(id)] = mentions
    pickle.dump(id2mentions_edit, open(id2mentions_path, mode="wb"))
    
    id2PMIDs_edit = {}
    for id, PMIDs in tqdm(id2PMIDs.items(), desc=f"processing the id2PMIDs_edit", total=len(id2PMIDs)):
        id2PMIDs_edit[int(id)] = PMIDs
    
    
    pickle.dump(id2PMIDs_edit, open(id2PMIDs_path, mode="wb"))
        
    
def temp_step6_again():
    # Step 6: Retrieve all the triples, for each triple (s, r, o), find the coexist PMIDs for s and o.
    save_snomedCT_data_path = "/nfs/long_tail/my_datasets/usml/SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z/process"
    entity2id_save_path = os.path.join(save_snomedCT_data_path, "ent2id.pickle")
    id2entityname_save_path = os.path.join(save_snomedCT_data_path, "id2entityname.pickle")

    triples_save_path = os.path.join(save_snomedCT_data_path, "triples.pickle")
    triples = pickle.load(open(triples_save_path, mode="rb"))
    triples = set(triples)
    
    save_data_path = "/nfs/long_tail/entity_linking_preprocess"
    id2PMIDs_path = os.path.join(save_data_path, f"{file_name}_id2PMIDs.pickle")
    id2PMIDs = pickle.load(open(id2PMIDs_path, mode="rb"))
    
    ent2id = pickle.load(open(entity2id_save_path, mode="rb"))
    n_triples = len(triples)
    print(f"Step 6: Retrieve all the triples, for each triple (s, r, o) in total {n_triples} triples, find the coexist PMIDs for s and o.")
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


# ============================ #
#         Sentence Level       #
# ============================ #

def generate_PMID2PubMedAbstract_data():
    # data = extarct_PubMed_data_via_PMID(17299597)
    file_name_list = ['cellline2pubtatorcentral', 'chemical2pubtatorcentral', 'disease2pubtatorcentral', 'gene2pubtatorcentral', 'mutation2pubtatorcentral', 'species2pubtatorcentral']    
    multi_pub_processor = MultiPubtatorProcessor(file_name_list)
    file_name = multi_pub_processor.merged_file_name
    save_data_path = "/nfs/long_tail/entity_linking_preprocess"
    save_snomedCT_data_path = "/nfs/long_tail/my_datasets/usml/SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z/process"
    triples_save_path = os.path.join(save_snomedCT_data_path, "triples.pickle")
    triples = pickle.load(open(triples_save_path, mode="rb"))
    triples = set(triples)
    n_triples = len(triples)
    
    triple2coexistPMIDs_path = os.path.join(save_data_path, f"{file_name}_triple2coexistPMIDs.pickle")
    triple2coexistPMIDs = pickle.load(open(triple2coexistPMIDs_path, mode="rb"))
    
    all_coexistPMIDs = set()
    
    for triple in tqdm(triples, desc="Scanning all the triples ...", total=n_triples):
        coexistPMIDs = triple2coexistPMIDs[triple]
        all_coexistPMIDs.update(coexistPMIDs)
        
    print("Successfully scanning all the triples!")
    
    PMID2PubMedAbstract = {}
    
    for PMID in tqdm(all_coexistPMIDs, desc="Downloading PMID-related PubMed abstract online ...", total=len(all_coexistPMIDs)):
        """
            data = {'PMID': PMID, 'title': title, 'abstract': abstract}
        """
        data = extarct_PubMed_data_via_PMID(PMID)
        PMID2PubMedAbstract[PMID] = data
    
    print("Saving PMID2PubMedAbstract ...")
    pickle.dump(PMID2PubMedAbstract, open(os.path.join(save_data_path, f"{file_name}_PMID2PubMedAbstract.pickle"), mode="wb"))
    
    final_save_path = os.path.join(save_data_path, f"{file_name}_PMID2PubMedAbstract.pickle")
    print(
        f"Successfully Saving PMID2PubMedAbstract to {final_save_path}"
        )
    
    
    

       
    
if __name__ == "__main__":
    # print(5 * (10 ** 6))
    # results = [
    #     [('168113', 0.69278115), ('352272', 0.61756897), ('217145', 0.6094995)], 
    #     [('65713', 0.68059087), ('340646', 0.67876977), ('364605', 0.67388695)], 
    #     [('127568', 0.6484056), ('127541', 0.63720894), ('191111', 0.63089496)], 
    #     [('173760', 0.99999994), ('173765', 0.9921722), ('95654', 0.89093053)], 
    #     [('433799', 0.86873823), ('92394', 0.8502206), ('293822', 0.8179185)], 
    #     [('348926', 0.84749866), ('99038', 0.8402275), ('350606', 0.8162697)], 
    #     [('191388', 0.9277023), ('16027', 0.91782147), ('16028', 0.90698355)], 
    #     [('91902', 0.9999999), ('92501', 0.9762863), ('181433', 0.899636)], 
    #     [('153945', 0.9005642), ('109368', 0.883803), ('263214', 0.8809234)], 
    #     [('160324', 1.0), ('48288', 0.9796798), ('160312', 0.84929454)]
    # ]
    
    # outputs = []
    # for topk_str_indices, topk_score_array in results:
    #     outputs.append([(idx, score) for idx, score in zip(topk_str_indices, topk_score_array)])
    
    # test_mention_list = ["william bradford", "governor", "Disneyland", "blood diseases", "sexually transmitted", "cardiopathy", "covid-19", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"]
    
    # sapbert_entity_linking(test_mention_list)
    
    # from pdb import set_trace; set_trace()
    # a = 0
    
    #! release
    # print("Extarct the \"triples\" from usml > SnomedCT.")
    # save_usml_data_path = "/nfs/long_tail/my_datasets/usml"
    # triples_save_path = os.path.join(save_usml_data_path, "triples.pickle")
    # triples = pickle.load(open(triples_save_path, mode="rb"))
    
    # triples_set = set(triples)
    
    # print(f"n_triples is {len(triples)}")
    # print(f"n_triples without duplication is {len(triples_set)}")
    
    
    # main()
    
    
    # ========================== #
    #          Statistics        #
    # ========================== #
    file_name_list = ['cellline2pubtatorcentral', 'chemical2pubtatorcentral', 'disease2pubtatorcentral', 'gene2pubtatorcentral', 'mutation2pubtatorcentral', 'species2pubtatorcentral']
    # pub_processor = PubtatorProcessor('mutation2pubtatorcentral')
    
    multi_pub_processor = MultiPubtatorProcessor(file_name_list)
    file_name = multi_pub_processor.merged_file_name
    
    # file_name = pub_processor.file_name
    # statistics_id2mentions(file_name)
    # statistics_id2PMIDs(file_name)
    # statistics_triple2coexistPMIDs(file_name)
    # statistics_count_id_from_tripple_in_id2mentions(file_name)
    
    # temp_method_convert_str2int_wihin_dict()
    # temp_step6_again()
    
    
    # ==================================  #
    #         Sentence Level Process      #
    # =================================== #
    # generate_PMID2PubMedAbstract_data()