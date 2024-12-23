import sys

sys.path.append("/nfs")
sys.path.append("/nfs/long_tail")
sys.path.append("/nfs/general")

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle

# from long_tail.my_datasets.pubtator.pubtator_processor import PubtatorProcessor, MultiPubtatorProcessor
from data.pubtator.pubtator_processor import PubtatorProcessor, MultiPubtatorProcessor

from utils.my_utils import load_json, save_json


save_data_path = "/nfs/long_tail/entity_linking_preprocess"
    
save_snomedCT_data_path = "/nfs/long_tail/my_datasets/usml/SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z/process"

file_name_list = ['cellline2pubtatorcentral', 'chemical2pubtatorcentral', 'disease2pubtatorcentral', 'gene2pubtatorcentral', 'mutation2pubtatorcentral', 'species2pubtatorcentral']
# pub_processor = PubtatorProcessor('mutation2pubtatorcentral')
multi_pub_processor = MultiPubtatorProcessor(file_name_list)
file_name = multi_pub_processor.merged_file_name


class LoadPreprocessData:
    @staticmethod
    def load_triples():
        triples_save_path = os.path.join(save_snomedCT_data_path, "triples.pickle")
        triples = pickle.load(open(triples_save_path, mode="rb"))
        return triples
    
    @staticmethod
    def load_entity2id():
        entity2id_save_path = os.path.join(save_snomedCT_data_path, "ent2id.pickle")
        ent2id = pickle.load(open(entity2id_save_path, mode="rb"))
        return ent2id
    
    @staticmethod
    def load_id2entityname():
        id2entityname_save_path = os.path.join(save_snomedCT_data_path, "id2entityname.pickle")
        id2entityname = pickle.load(open(id2entityname_save_path, mode="rb"))
        return id2entityname
    
    @staticmethod
    def load_relation2id():
        relation2id_save_path = os.path.join(save_snomedCT_data_path, "relation2id.pickle")
        relation2id = pickle.load(open(relation2id_save_path, mode="rb"))
        return relation2id
    
    @staticmethod
    def load_id2relationname():
        id2relationname_save_path = os.path.join(save_snomedCT_data_path, "id2relationname.pickle")
        id2relationname = pickle.load(open(id2relationname_save_path, mode="rb"))
        return id2relationname
    
    @staticmethod
    def load_id2mentions():
        id2mentions_path = os.path.join(save_data_path, f"{file_name}_id2mentions.pickle")
        id2mentions = pickle.load(open(id2mentions_path, mode="rb"))
        return id2mentions
    
    @staticmethod
    def load_id2midxs():
        # midx 是 mention index
        id2midxs_path = os.path.join(save_data_path, f"{file_name}_id2midxs.pickle")
        # id2midxs这里面id是字符串，midx是数字
        id2midxs = pickle.load(open(id2midxs_path, mode="rb"))
        return id2midxs
    
    @staticmethod
    def load_triple2coexistPMIDs():
        triple2coexistPMIDs_path = os.path.join(save_data_path, f"{file_name}_triple2coexistPMIDs.pickle")
        triple2coexistPMIDs = pickle.load(open(triple2coexistPMIDs_path, mode="rb"))
        return triple2coexistPMIDs
    
    @staticmethod
    def load_triple2coexist_num():
        triple2coexistPMIDs = LoadPreprocessData.load_triple2coexistPMIDs()
        def convert_triple2coexistPMIDs_to_triple2coexist_num(triple2coexistPMIDs: dict):
            triple2coexist_num = {}
            for triple, PMIDs in triple2coexistPMIDs.items():
                triple2coexist_num[triple] = len(PMIDs)
            return triple2coexist_num
        triple2coexist_num = convert_triple2coexistPMIDs_to_triple2coexist_num(triple2coexistPMIDs)
        return triple2coexist_num
    
    @staticmethod
    def load_relation_data(is_filter=True, related_triple_num_threshold=1000):
        relation_data_path = os.path.join(save_data_path, 'relation_data.json')
        relation_data = load_json(relation_data_path)
        if is_filter:
            old_relation_num = len(relation_data)
            relation_data = [single_relation_data for single_relation_data in relation_data if single_relation_data['num'] > related_triple_num_threshold]
            new_relation_num = len(relation_data)
            print(f"Filter algorithm is ON, the original relation number is {old_relation_num}, the filter relation number is {new_relation_num}")
            
        return relation_data

class LoadPromptData:
    
    prompt_data_save_path = "/nfs/long_tail/knowledge_probing"
    
    @staticmethod
    def load_filtered_relations_with_prompt_data():
        filtered_relations_with_prompt_data_save_path = os.path.join(LoadPromptData.prompt_data_save_path, "filtered_relations_with_prompt_data.json")
        filtered_relations_with_prompt_data = load_json(filtered_relations_with_prompt_data_save_path)
        return filtered_relations_with_prompt_data
    
    @staticmethod
    def load_filtered_relations2prompt():
        relations_with_prompt_data_save_path = os.path.join(LoadPromptData.prompt_data_save_path, "filtered_relations_with_prompt_data.json")
        relations_with_prompt_data = load_json(relations_with_prompt_data_save_path)
        filtered_relation2prompt = {}
        for single_relation_with_prompt in relations_with_prompt_data:
            relation = single_relation_with_prompt['relation']
            prompt = single_relation_with_prompt['prompt']
            filtered_relation2prompt[relation] = prompt
        return filtered_relation2prompt

