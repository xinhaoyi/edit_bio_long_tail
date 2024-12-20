import sys
from typing import Dict, List, Tuple, Union

# sys.path.append("/nfs")
# sys.path.append("/nfs/general")
# sys.path.append("/nfs/long_tail")

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data.pubtator.pubtator_processor import MultiPubtatorProcessor
from dsets.pubtator_dataset import PubTatorDataset
from dsets.knowledge_graph_dataset import KnowledgeGraphDataset, SnomedCTDataset, UMLSDataset
from utils.my_utils import load_json, random_state, save_json
import yaml
# from parrot import Parrot
import torch
import warnings
from config import path_config

# def gen_default_parrot():
#     random_state(42)
#     parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
#     return parrot

# default_parrot = gen_default_parrot()

root_path = str(path_config.BASE_DIR)

default_save_path = path_config.KNOWLEDGE_TRIPLES_DATA_DIR

class SnomedCTPromptDataset:
    def __init__(self, prompt_mode: int, save_path: str=default_save_path) -> None:
        self.__save_path = save_path
        self.relation_name2k_triples_with_name = self.__get_relation_name2k_triples_with_name()
        __relationname2prompt_template_list = self.__load_relationname2prompt_template_list()
        
        __n_relationname2prompt_template_list = len(__relationname2prompt_template_list)
        
        self.prompt_mode = prompt_mode % __n_relationname2prompt_template_list
        
        self.selected_relationname2prompt_template = __relationname2prompt_template_list[self.prompt_mode]
        
        # self.rephrase_parrot = rephrase_parrot
        # para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False)
                
        # self.filtered_pubtator_linked_to_snomedCT_dataset = FilterPubtatorLinkedToSnomedCTDataset()
        # FilterPubtatorLinkedToSnomedCTDataset.load_triple_with_real_name2triple()
        # load_relation_name2k_triples_with_real_name
    
    def transform_triple_with_real_name_to_prompt(self, triple_with_real_name, relationname2prompt_template=None):
        head_entity_name = triple_with_real_name[0]
        relation_name = triple_with_real_name[1]
        # tail_entity_name = triple_with_real_name[2]
        if relationname2prompt_template is None or len(relationname2prompt_template) == 0:
            inner_relationname2prompt_template = self.selected_relationname2prompt_template
        else:
            inner_relationname2prompt_template = relationname2prompt_template
            
        if relation_name in inner_relationname2prompt_template.keys():
            # example of relationname2prompt_template: Finding site: "What is the finding site of {head_entity_name}?"
            prompt_template = inner_relationname2prompt_template[relation_name]
            prompt = prompt_template.format(head_entity_name=head_entity_name)
        else:
            # '[DEFAULT_RELATION]': "What is the {defualt_relation} of {head_entity_name}?"
            try:
                prompt_template = inner_relationname2prompt_template['[DEFAULT_RELATION]']
                prompt = prompt_template.format(default_relation=relation_name, head_entity_name=head_entity_name)
            except:
                # from pdb import set_trace; set_trace()
                raise Exception(f'We can\'t tranform the {triple_with_real_name} to prompt with current template: {prompt_template}')
                
        return prompt
    
    def transform_triple_with_real_name_list_to_prompt_list(self, triple_with_real_name_list):
        prompt_list = []
        for triple_with_real_name in triple_with_real_name_list:
            prompt = self.transform_triple_with_real_name_to_prompt(triple_with_real_name)
            prompt_list.append(prompt)
        return prompt_list
    
    
    def transform_triple_with_real_name_to_prompt_with_demonstration(self, triple_with_real_name, relationname2prompt_template=None):
        relation_name = triple_with_real_name[1]
        k_demonstration = self.__get_k_demonstration_for_relation_name(relation_name)
        prompt = self.transform_triple_with_real_name_to_prompt(triple_with_real_name, relationname2prompt_template)
        
        prompt_with_demonstration = f"{k_demonstration}{prompt}"
        
        return prompt_with_demonstration
    
    def transform_triple_with_real_name_list_to_prompt_with_demonstration_list(self, triple_with_real_name_list):
        prompt_with_demonstration_list = []
        for triple_with_real_name in triple_with_real_name_list:
            prompt_with_demonstration = self.transform_triple_with_real_name_to_prompt_with_demonstration(triple_with_real_name)
            prompt_with_demonstration_list.append(prompt_with_demonstration)
        return prompt_with_demonstration_list
    
    def generate_rephrase_prompt(self, original_prompt: str, triple_with_real_name, rephrase_template_mode: int = 0, is_print: bool=False, is_demonstration: bool=False):
        # para_phrases = self.rephrase_parrot.augment(input_phrase=original_prompt, use_gpu=False)
        # rephrase_prompt = para_phrases[0][0] if len(para_phrases) > 0 else None
        # todo
        relationname2prompt_template_list = self.__load_relationname2prompt_template_list_for_rephrase()
        relationname2prompt_template = relationname2prompt_template_list[rephrase_template_mode]
        if not is_demonstration:
            rephrase_prompt = self.transform_triple_with_real_name_to_prompt(triple_with_real_name=triple_with_real_name, relationname2prompt_template=relationname2prompt_template)
        else:
            rephrase_prompt = self.transform_triple_with_real_name_to_prompt_with_demonstration(triple_with_real_name=triple_with_real_name, relationname2prompt_template=relationname2prompt_template)
        target = triple_with_real_name[2]
        if is_print:
            print("Preparing rephrase prompt ...")
            print("The original prompt is: ")
            print(original_prompt)
            print("The rephrase prompt:")
            print(rephrase_prompt)
        return rephrase_prompt
    
    def generate_locality_prompt_and_answer(self, triple_with_real_name, is_demonstration: bool=False):        
        if not is_demonstration:
            prompt = self.transform_triple_with_real_name_to_prompt(triple_with_real_name=triple_with_real_name)
        else:
            prompt = self.transform_triple_with_real_name_to_prompt_with_demonstration(triple_with_real_name=triple_with_real_name)
        tail_entity = triple_with_real_name[-1]
        answer = tail_entity
        
        return prompt, answer
    
    def generate_portability_prompt_and_answer(self, triple_with_real_name):
        pass
        
    
    def __get_relation_name2k_triples_with_name(self):
        relation_name2k_triples_with_name_file_path = os.path.join(root_path, 'data', 'pubtator2snomedCT', 'relation_name2k_triples_with_real_name.json')
        relation_name2k_triples_with_name = load_json(relation_name2k_triples_with_name_file_path)
        
        relation_name2k_triples_with_name_processed = {}
        
        for relation_name, k_triples_with_name in relation_name2k_triples_with_name.items():
            k_triples_with_name_list = []
            # tranform form list to tuple
            for triple_with_name in k_triples_with_name:
                triple_with_name = (triple_with_name[0], triple_with_name[1], triple_with_name[2])
                k_triples_with_name_list.append(triple_with_name)
            relation_name2k_triples_with_name_processed[relation_name] = k_triples_with_name_list
        
        return relation_name2k_triples_with_name_processed
    
    def __get_k_demonstration_for_relation_name(self, relation_name):
        k_triples_with_name = self.relation_name2k_triples_with_name[relation_name]
        prompt_list = self.transform_triple_with_real_name_list_to_prompt_list(k_triples_with_name)
        
        k_demonstration = ""
        
        for triple_with_name, prompt in zip(k_triples_with_name, prompt_list):
            answer = triple_with_name[2]
            prompt = f"{prompt}\n{answer}\n\n"
            k_demonstration += prompt
        
        return k_demonstration
        

    def __load_relationname2prompt_template_list(self) -> List[Dict[str, str]]:
        relation_name2prompt_template_file_name = "relation_name2prompt_template.yaml"
        relation_name2prompt_template_file_path = os.path.join(self.__save_path, relation_name2prompt_template_file_name)
        
        with open(relation_name2prompt_template_file_path, 'r') as file:
            """
            keys: relationname2prompt_template_0, relationname2prompt_template_1, relationname2prompt_template_2 ...
            """
            relation_name2prompt_template_dict =  yaml.safe_load(file)
            relationname2prompt_template_list = list(relation_name2prompt_template_dict.values())
            
            # print(f"{len(relationname2prompt_template_list)}")
            # print(relationname2prompt_template_list[0])
            
            return relationname2prompt_template_list
        
    def __load_relationname2prompt_template_list_for_rephrase(self) -> List[Dict[str, str]]:
        relation_name2prompt_template_file_name = "relation_name2prompt_template_for_rephrase.yaml"
        relation_name2prompt_template_file_path = os.path.join(self.__save_path, relation_name2prompt_template_file_name)
        with open(relation_name2prompt_template_file_path, 'r') as file:
            """
            keys: relationname2prompt_template_0, relationname2prompt_template_1, relationname2prompt_template_2 ...
            """
            relation_name2prompt_template_dict =  yaml.safe_load(file)
            relationname2prompt_template_list = list(relation_name2prompt_template_dict.values())
            
            # print(f"{len(relationname2prompt_template_list)}")
            # print(relationname2prompt_template_list[0])
            
            return relationname2prompt_template_list
        
    
if __name__ == "__main__":
    # prompt_mode = 0-4
    snomedCT_prompt_dataset = SnomedCTPromptDataset(prompt_mode=2)
    test_data = ('Ulnar neuropathy', 'Finding site', 'Ulnar nerve')
    print(test_data)
    print("********************************")
    prompt = snomedCT_prompt_dataset.transform_triple_with_real_name_to_prompt(test_data)
    print(prompt)
    print("********************************")
    prompt_with_demonstration = snomedCT_prompt_dataset.transform_triple_with_real_name_to_prompt_with_demonstration(test_data)
    print(prompt_with_demonstration)
    
