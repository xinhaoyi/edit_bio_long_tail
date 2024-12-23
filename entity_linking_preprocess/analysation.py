import sys

from data.pubtator.pubtator_processor import MultiPubtatorProcessor
sys.path.append("/nfs/long_tail")
sys.path.append("/nfs/general")

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from entity_linking_preprocess.load_data import LoadPreprocessData
from utils.my_utils import load_json, save_json
import pickle
from tqdm import tqdm

triple2coexist_num = LoadPreprocessData.load_triple2coexist_num


def statistics_id2mentions(file_name):
    print(f"statistics id2mentions dictionary ...")
    save_data_path = "/nfs/long_tail/entity_linking_preprocess"
    id2mentions_path = os.path.join(save_data_path, f"{file_name}_id2mentions.pickle")
    id2mentions = pickle.load(open(id2mentions_path, mode="rb"))

    n_id = len(id2mentions)
    mentions_set = set()
    for id, mentions in id2mentions.items():
        if len(mentions) > 0:
            mentions_set.update(mentions)
    
    n_mentions = len(mentions_set)
    
    print(f"n_id: {n_id}")
    print(f"n_mentions: {n_mentions}")
    
def statistics_id2PMIDs(file_name):
    print(f"statistics id2PMIDs dictionary ...")
    save_data_path = "/nfs/long_tail/entity_linking_preprocess"
    id2PMIDs_path = os.path.join(save_data_path, f"{file_name}_id2PMIDs.pickle")
    id2PMIDs = pickle.load(open(id2PMIDs_path, mode="rb"))
    
    n_id = len(id2PMIDs)
    PMIDs_set = set()
    for id, PMIDs in id2PMIDs.items():
        if len(PMIDs) > 0:
            PMIDs_set.update(PMIDs)
    
    n_PMIDs = len(PMIDs_set)
    
    print(f"n_id: {n_id}")
    print(f"n_PMIDs: {n_PMIDs}")
    
    
def statistics_triple2coexistPMIDs(file_name):
    print(f"statistics triple2coexistPMIDs dictionary ...")
    save_data_path = "/nfs/long_tail/entity_linking_preprocess"
    triple2coexistPMIDs_path = os.path.join(save_data_path, f"{file_name}_triple2coexistPMIDs.pickle")
    triple2coexistPMIDs = pickle.load(open(triple2coexistPMIDs_path, mode="rb"))
    
    save_snomedCT_data_path = "/nfs/long_tail/my_datasets/usml/SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z/process"
    entity2id_save_path = os.path.join(save_snomedCT_data_path, "ent2id.pickle")
    ent2id = pickle.load(open(entity2id_save_path, mode="rb"))
    
    id2entityname_save_path = os.path.join(save_snomedCT_data_path, "id2entityname.pickle")
    id2entityname = pickle.load(open(id2entityname_save_path, mode="rb"))
    
    n_triples = len(triple2coexistPMIDs.keys())
    n_triples_with_coexistPMIDs = 0
    coexistPMIDs_set = set()
    for triple, coexistPMIDs in triple2coexistPMIDs.items():
        if len(coexistPMIDs) > 0:
            n_triples_with_coexistPMIDs += 1
            coexistPMIDs_set.update(coexistPMIDs)
    
    n_coexistPMIDs = len(coexistPMIDs_set)
    
    print(f"n_triples: {n_triples}")
    print(f"n_triples_with_coexistPMIDs: {n_triples_with_coexistPMIDs}")
    print(f"n_coexistPMIDs: {n_coexistPMIDs}")
    
    def convert_triple2coexistPMIDs_to_triple2coexist_num(triple2coexistPMIDs: dict):
        triple2coexist_num = {}
        for triple, PMIDs in triple2coexistPMIDs.items():
            triple2coexist_num[triple] = len(PMIDs)
        return triple2coexist_num
    
    triple2coexist_num = convert_triple2coexistPMIDs_to_triple2coexist_num(triple2coexistPMIDs)
    
    filtered_triples = {triple: num for triple, num in triple2coexist_num.items() if num > 46000}

    # 按照 coexist_num 大小从大到小排序
    sorted_triples = sorted(filtered_triples.items(), key=lambda x: x[1], reverse=True)
    
    output_dict = {(id2entityname[ent2id[sorted_triple[0]]], sorted_triple[1], id2entityname[ent2id[sorted_triple[2]]]) : num for sorted_triple, num in sorted_triples}
    
    print(output_dict)
    
    
def statistics_count_id_from_tripple_in_id2mentions(file_name):
    # print(f"Counting id from tripples in id2mentions ...")
    save_data_path = "/nfs/long_tail/entity_linking_preprocess"
    
    save_snomedCT_data_path = "/nfs/long_tail/my_datasets/usml/SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z/process"
    triples_save_path = os.path.join(save_snomedCT_data_path, "triples.pickle")
    triples = pickle.load(open(triples_save_path, mode="rb"))
        
    entity2id_save_path = os.path.join(save_snomedCT_data_path, "ent2id.pickle")
    ent2id = pickle.load(open(entity2id_save_path, mode="rb"))
    
    id2mentions_path = os.path.join(save_data_path, f"{file_name}_id2mentions.pickle")
    id2mentions = pickle.load(open(id2mentions_path, mode="rb"))
    
    id2midxs_path = os.path.join(save_data_path, f"{file_name}_id2midxs.pickle")
    
    # id2midxs这里面id是字符串，midx是数字
    id2midxs = pickle.load(open(id2midxs_path, mode="rb"))
    
    id2entityname_save_path = os.path.join(save_snomedCT_data_path, "id2entityname.pickle")
    id2entityname = pickle.load(open(id2entityname_save_path, mode="rb"))
    
    
    # s_exist_triples = []
    # o_exist_triples = []
    # both_exist_triples = []
    
    # for triple in triples:
    #     s_eid = triple[0]
    #     # r_id = triple[1]
    #     o_eid = triple[2]
                
    #     s_id = ent2id[s_eid]
    #     o_id = ent2id[o_eid]
        
    #     s_name = id2entityname[s_id]
    #     o_name = id2entityname[o_id]
        
    #     if s_id in id2mentions and o_id in id2mentions:
    #         both_exist_triples.append((triple, s_eid, o_eid))
    #     elif s_id in id2mentions:
    #         s_exist_triples.append((triple, s_eid))
    #     elif o_id in id2mentions:
    #         o_exist_triples.append((triple, o_eid))
    
    # ========================== #
    #       Mini batch Test      #
    # ========================== #
    sub_triples = triples[:10]
    sub_ids = list(id2mentions.keys())[:10]
    print("sub ids:")
    print(sub_ids)
    
    # sub_midxs = list(id2midxs.values())[:10]
    # print("sub midxs:")
    # print(sub_midxs)
    
    sub_idxs_from_id2midxs = list(id2midxs.keys())[:10]
    print("sub_idxs_from_id2midxs:")
    print(sub_idxs_from_id2midxs)
    
    # sub_ids_from_ent2id = list(ent2id.values())[:10]
    # print("sub_ids_from_ent2id:")
    # print(sub_ids_from_ent2id)
    
    # sub_ids_from_id2entityname = list(id2entityname.keys())[:10]
    # print("sub_ids_from_id2entityname")
    # print(sub_ids_from_id2entityname)
    
    
    # test_list = []
    # for triple in sub_triples:
    #     s_eid = triple[0]
    #     # r_id = triple[1]
    #     o_eid = triple[2]
                
    #     s_id = ent2id[s_eid]
    #     o_id = ent2id[o_eid]
        
    #     s_name = id2entityname[s_id]
    #     o_name = id2entityname[o_id]
        
    #     test_list.append((s_eid, s_id, s_name, o_eid, o_id, o_name))
    
    # print(test_list)
    
    # ========================== #
    #         Print Result       #
    # ========================== #
    print(f"Num of tripples: {len(set(triples))}")
    print(f"Num of entity_ids: {len(id2mentions.keys())}")
    
    mentions_set = set()
    for id, mentions in id2mentions.items():
        if len(mentions) > 0:
            mentions_set.update(mentions)
    print(f"Num of mentions: {len(mentions_set)}")
        
    # print("subject of the triples exists in id2mention......")
    # print(len(s_exist_triples))
    
    # print("object of the triples exists in id2mention......")
    # print(len(o_exist_triples))
    
    # print("subject and object of the triples exists in id2mention......")
    # print(len(both_exist_triples))

def statistics_triple2coexistPMIDs():
    print(f"statistics triple2coexistPMIDs dictionary ...")
    triple2coexistPMIDs = LoadPreprocessData.load_triple2coexistPMIDs()
    ent2id = LoadPreprocessData.load_entity2id()
    id2entityname = LoadPreprocessData.load_id2entityname()
    
    relation2id = LoadPreprocessData.load_relation2id()
    id2relationname = LoadPreprocessData.load_id2relationname()
    
    n_triples = len(triple2coexistPMIDs.keys())
    n_triples_with_coexistPMIDs = 0
    coexistPMIDs_set = set()
    for triple, coexistPMIDs in triple2coexistPMIDs.items():
        if len(coexistPMIDs) > 0:
            n_triples_with_coexistPMIDs += 1
            coexistPMIDs_set.update(coexistPMIDs)
    
    n_coexistPMIDs = len(coexistPMIDs_set)
    
    print(f"n_triples: {n_triples}")
    print(f"n_triples_with_coexistPMIDs: {n_triples_with_coexistPMIDs}")
    print(f"n_coexistPMIDs: {n_coexistPMIDs}")
    
    
    triple2coexist_num = LoadPreprocessData.load_triple2coexist_num()
    
    filtered_triples = {triple: num for triple, num in triple2coexist_num.items() if num > 46000}

    # 按照 coexist_num 大小从大到小排序
    sorted_triples = sorted(filtered_triples.items(), key=lambda x: x[1], reverse=True)
    
    output_dict = {(id2entityname[ent2id[sorted_triple[0]]], id2relationname[relation2id[sorted_triple[1]]], id2entityname[ent2id[sorted_triple[2]]]) : num for sorted_triple, num in sorted_triples}
    
    print(output_dict)



def statistics_count_id_from_tripple_in_id2mentions():
    file_name_list = ['cellline2pubtatorcentral', 'chemical2pubtatorcentral', 'disease2pubtatorcentral', 'gene2pubtatorcentral', 'mutation2pubtatorcentral', 'species2pubtatorcentral']    
    multi_pub_processor = MultiPubtatorProcessor(file_name_list)
    file_name = multi_pub_processor.merged_file_name
    # print(f"Counting id from tripples in id2mentions ...")
    save_data_path = "/nfs/long_tail/entity_linking_preprocess"
    
    save_snomedCT_data_path = "/nfs/long_tail/my_datasets/usml/SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z/process"
    triples_save_path = os.path.join(save_snomedCT_data_path, "triples.pickle")
    triples = pickle.load(open(triples_save_path, mode="rb"))
    triples = set(triples)
    n_triples = len(triples)
        
    entity2id_save_path = os.path.join(save_snomedCT_data_path, "ent2id.pickle")
    ent2id = pickle.load(open(entity2id_save_path, mode="rb"))
    
    id2mentions_path = os.path.join(save_data_path, f"{file_name}_id2mentions.pickle")
    id2mentions = pickle.load(open(id2mentions_path, mode="rb"))
    
    id2midxs_path = os.path.join(save_data_path, f"{file_name}_id2midxs.pickle")
    
    # id2midxs这里面id是字符串，midx是数字 
    id2midxs = pickle.load(open(id2midxs_path, mode="rb"))
    
    id2entityname_save_path = os.path.join(save_snomedCT_data_path, "id2entityname.pickle")
    id2entityname = pickle.load(open(id2entityname_save_path, mode="rb"))
    
    triple2coexistPMIDs_path = os.path.join(save_data_path, f"{file_name}_triple2coexistPMIDs.pickle")
    triple2coexistPMIDs = pickle.load(open(triple2coexistPMIDs_path, mode="rb"))
    
    for triple in tqdm(triples, desc="Scanning all the triples ...", total=n_triples):
        coexistPMIDs = triple2coexistPMIDs[triple]
        s_eid = triple[0]
        # r_id = triple[1]
        o_eid = triple[2]
        
        # from pdb import set_trace; set_trace()
        
        sid = ent2id[s_eid]
        oid = ent2id[o_eid]
        
        
        s_mentions = id2mentions[sid]
        o_mentions = id2mentions[oid]
        
        s_entity_name = id2entityname[sid]
        o_entity_name = id2entityname[oid]



"""
Conventional release intraarticular implant has dose form intended site intraarticular
"""

        
    
if __name__ == "__main__":
    statistics_triple2coexistPMIDs()