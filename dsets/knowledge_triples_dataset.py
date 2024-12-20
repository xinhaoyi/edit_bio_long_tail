import random
import sys
from typing import Dict, List, Optional, Tuple

# sys.path.append("/nfs")
# sys.path.append("/nfs/general")
# sys.path.append("/nfs/long_tail")

import torch
from torch.utils.data import Dataset
# from parrot import Parrot

from data.pubtator.pubtator_processor import MultiPubtatorProcessor
from dsets.pubtator_dataset import PubTatorDataset
from dsets.prompt_dataset import SnomedCTPromptDataset

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils.my_utils import load_json, save_json, random_state

from config import path_config

default_save_knowledge_triples_data_path = str(path_config.KNOWLEDGE_TRIPLES_DATA_DIR)
# default_save_knowledge_triples_data_path = "/data/knowledge_triples"
default_save_data_file_name = str(path_config.KNOWLEDGE_TRIPLES_RAW_DATA_FILE_NAME)

class SnomedCTKnowledgeTriplesDataset(Dataset):
    def __init__(self, 
                 frequency_type: str,
                 save_data_path: str = default_save_knowledge_triples_data_path,
                 save_data_file_name: str = default_save_data_file_name,
                 is_1_to_1_triples = True,
                 is_1_to_N_triples = True,
                 is_N_to_1_triples = True,
                 is_N_to_N_triples = True,
                 frequency_dict = None,
                 is_sample=True,
                 random_select_datasize = None
                 ) -> None:   
        self.frequency_type = frequency_type
        self.save_data_path = save_data_path
        self.save_data_file_name = save_data_file_name
        if frequency_dict is None or len(frequency_dict) == 0:
            self.frequency_dict = {
                'long_tail': (0,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None)
            }
            # self.frequency_dict = {
            #     'long_tail': (0,1),
            #     'medium': (1,10),
            #     'high': (10, 100),
            #     'very_high': (100, None)
            # }
        else:
            self.frequency_dict = frequency_dict
        if frequency_type is not None and (frequency_type in list(self.frequency_dict.keys()) or frequency_type in ['all', '']):
            self.umls_knowledge_triples_data = self.get_umls_knowledge_triples_data(frequency_type=self.frequency_type,
                                                                 is_1_to_1_triples=is_1_to_1_triples,
                                                                 is_1_to_N_triples=is_1_to_N_triples,
                                                                 is_N_to_1_triples=is_N_to_1_triples,
                                                                 is_N_to_N_triples=is_N_to_N_triples
                                                                 )
            # ! delete
            # self.umls_knowledge_triples_data = self.umls_knowledge_triples_data[:10]
            
            if random_select_datasize != None:
                random.seed(42)
                if len(self.umls_knowledge_triples_data) < random_select_datasize:
                    random_select_datasize = len(self.umls_knowledge_triples_data)
                selected_umls_knowledge_triples_data = random.sample(self.umls_knowledge_triples_data, random_select_datasize)
                self.umls_knowledge_triples_data = selected_umls_knowledge_triples_data
            
        else:
            self.umls_knowledge_triples_data = []
        
        if is_sample:
            self.knowledge_triples_data_1_1, self.knowledge_triples_data_1_N, self.knowledge_triples_data_N_1, self.knowledge_triples_data_N_N = self.get_knowledge_triples_data_based_on_relationship_type(umls_knowledge_triples_data=self.umls_knowledge_triples_data, is_sample=is_sample)
            
            self.data_1_1_indexes = [knowledge_triple_data['index'] for knowledge_triple_data in self.knowledge_triples_data_1_1] if len(self.knowledge_triples_data_1_1) > 0 else []
            self.data_1_N_indexes = [knowledge_triple_data['index'] for knowledge_triple_data in self.knowledge_triples_data_1_N] if len(self.knowledge_triples_data_1_N) > 0 else []
            self.data_N_1_indexes = [knowledge_triple_data['index'] for knowledge_triple_data in self.knowledge_triples_data_N_1] if len(self.knowledge_triples_data_N_1) > 0 else []
            self.data_N_N_indexes = [knowledge_triple_data['index'] for knowledge_triple_data in self.knowledge_triples_data_N_N] if len(self.knowledge_triples_data_N_N) > 0 else []
        
    def __len__(self):
        # ! delete
        # return len(self.umls_knowledge_triples_data[:10])
        return len(self.umls_knowledge_triples_data)
    
    def __getitem__(self, index):
        # data = self.umls_knowledge_triples_data[:10]
        # return data[index]
        return self.umls_knowledge_triples_data[index]
    
    def __load_umls_knowledge_triples_data(self):
        """
        list of: 
        {
            "index": 0,
            "triple": [
                "Pain of jaw",
                "Finding site",
                "Jaw"
            ],
            "coexist_num": 94,
            "answers": [
                "Jaw"
            ]
        }
        """
        umls_knowledge_triples_data_path = os.path.join(self.save_data_path, self.save_data_file_name)
        umls_knowledge_triples_data = load_json(umls_knowledge_triples_data_path)
        
        for single_umls_knowledge_triples_data in umls_knowledge_triples_data:
            head_entity_name = single_umls_knowledge_triples_data['triple'][0]
            relation_name = single_umls_knowledge_triples_data['triple'][1]
            tail_entity_name = single_umls_knowledge_triples_data['triple'][2]
            
            single_umls_knowledge_triples_data['triple'] = (head_entity_name, relation_name, tail_entity_name)
        
        return umls_knowledge_triples_data
    
    def load_one_to_many_knowledge_list(self, knowledge_triples_data=None) -> Tuple[List[List[int]], List[List[Dict]]]:
        if knowledge_triples_data == None:
            knowledge_triples_data = self.umls_knowledge_triples_data
            # ""relationship_type": "N_to_N","
        # from pdb import set_trace; set_trace()
        head_entity_relation2one_to_many_knowledge_triples_data: Dict[Tuple, List[Dict]] = {}
        for single_knowledge_triple_data in knowledge_triples_data:
            # from pdb import set_trace; set_trace()
            if single_knowledge_triple_data["relationship_type"] == "N_to_N" or single_knowledge_triple_data["relationship_type"] == "1_to_N":
                head_entity = single_knowledge_triple_data["triple"][0]
                relation = single_knowledge_triple_data["triple"][1]
                head_entity_relation: Tuple = (head_entity, relation)
                
                # 如果字典中不存在该键，则创建一个空列表
                if head_entity_relation not in head_entity_relation2one_to_many_knowledge_triples_data:
                    head_entity_relation2one_to_many_knowledge_triples_data[head_entity_relation] = []
                
                # 添加新的知识三元组数据到对应的键中
                head_entity_relation2one_to_many_knowledge_triples_data[head_entity_relation].append(single_knowledge_triple_data)
        
        # from pdb import set_trace; set_trace()
        one_to_many_knowledge_triples_data_list: List[List[Dict]] = []
        for head_entity_relation, one_to_many_knowledge_triples_data in head_entity_relation2one_to_many_knowledge_triples_data.items():
            one_to_many_knowledge_triples_data_list.append(one_to_many_knowledge_triples_data)
        
        one_to_many_knowledge_triples_indexs_list: List[List[int]] = [[single_knowledge_triple_data["index"] for single_knowledge_triple_data in one_to_many_knowledge_triples_data] for one_to_many_knowledge_triples_data in one_to_many_knowledge_triples_data_list]
        
        return one_to_many_knowledge_triples_indexs_list, one_to_many_knowledge_triples_data_list
    
    def check_at_least_single_coest_num_ok(self, min_num: int=None, max_num: int=None, knowledge_triples_data: List[Dict]=None,):
        """
        In knowledge_triples_data, if at least one single_knowledge_triple_data meet [min, max).
        It should be True, else False
        """
        # [min, max)
        if min_num is None:
            min_num = 0  # 如果min_num未设置，则默认为0
        if max_num is None:
            max_num = float('inf')  # 如果max_num未设置，则设为无穷大
        if min_num > max_num:
            temp = min_num
            min_num = max_num
            max_num = temp

        if knowledge_triples_data == None:
            knowledge_triples_data = self.umls_knowledge_triples_data
        for single_knowledge_triple_data in knowledge_triples_data:
            if single_knowledge_triple_data["coexist_num"] >= min_num and single_knowledge_triple_data["coexist_num"] < max_num:
                continue
            else:
                return False
    
    def load_knowledge_triple_data_from_data_indexes(self, data_indexes: List[int], knowledge_triples_data=None):
        if knowledge_triples_data == None:
            knowledge_triples_data = self.umls_knowledge_triples_data
        
        selected_knowledge_triples_data = []
        
        data_index2position_idx = {}
        for position_idx, single_knowledge_triple_data in enumerate(knowledge_triples_data):
            data_index = single_knowledge_triple_data["index"]
            data_index2position_idx[data_index] = position_idx
        
        for single_data_index in data_indexes:
            position_idx = data_index2position_idx[single_data_index]
            single_knowledge_triple_data = knowledge_triples_data[position_idx]
            selected_knowledge_triples_data.append(single_knowledge_triple_data)
        
        return selected_knowledge_triples_data
        
    
    def load_one_to_one_knowledge(self, knowledge_triples_data=None) -> Tuple[List[int], List[Dict]]:
        if knowledge_triples_data == None:
            knowledge_triples_data = self.umls_knowledge_triples_data
        
        all_one_to_one_knowledge = []
        for single_knowledge_triple_data in knowledge_triples_data:
            if single_knowledge_triple_data["relationship_type"] == "1_to_1" or single_knowledge_triple_data["relationship_type"] == "N_to_1":
                all_one_to_one_knowledge.append(single_knowledge_triple_data)
        
        all_one_to_one_knowledge_indexes = [single_one_to_one_knowledge["index"] for single_one_to_one_knowledge in all_one_to_one_knowledge]
        
        
        return all_one_to_one_knowledge_indexes, all_one_to_one_knowledge
    
    
    def load_all_answers(self, knowledge_triples_data=None) -> List[str]:
        if knowledge_triples_data == None:
            knowledge_triples_data = self.umls_knowledge_triples_data
        
        all_answers = []
        for single_knowledge_triple_data in knowledge_triples_data:
            answer = single_knowledge_triple_data["triple"][-1]
            all_answers.append(answer)
            
        return all_answers
    
    def get_single_knowledge_triple_data_by_index(self, index: int, knowledge_triples_data=None) -> Optional[List[Dict]]:
        if knowledge_triples_data == None:
            knowledge_triples_data = self.umls_knowledge_triples_data
        
        for single_knowledge_triple_data in knowledge_triples_data:
            if single_knowledge_triple_data["triple"] == index:
                return single_knowledge_triple_data
        
        return None 
        
    
    def statistics(self, 
                   clikt_data: List[Dict]=None):
        if clikt_data is None:
            clikt_data = self.umls_knowledge_triples_data
        
        subject_entity_set = set()
        relation_entity_set = set()
        object_entity_set = set()
        
        for single_clikt_data in clikt_data:
            knowledge = single_clikt_data["triple"]
            subject_entity = knowledge[0]
            relation_entity = knowledge[1]
            object_entity = knowledge[2]
            
            subject_entity_set.add(subject_entity)
            relation_entity_set.add(relation_entity)
            object_entity_set.add(object_entity)
                
        print(f"The total num of triples is {len(clikt_data)}\n")
        print(f"The total num of subjects is {len(subject_entity_set)}\n")
        print(f"The total num of objects is {len(object_entity_set)}\n")
        print(f"The total num of relation is {len(relation_entity_set)}\n")
        
        print(f"The total num of the knowledge groups is {len(clikt_data)}\n")
        
        # if is_one_to_many:
        #     print("======The statistics of the frequency distributions of one-to-many triples=======")
        #     self.__statistics_clikt_data_group_list(clikt_data_group_list=one_to_many_data_group_list, is_relation_level_details=is_relation_level_details)
        
        # if is_many_to_one:
        #     print("======The statistics of the frequency distributions of many-to-one triples======")
        #     self.__statistics_clikt_data_group_list(clikt_data_group_list=many_to_one_data_group_list, is_relation_level_details=is_relation_level_details)
        
    
    def get_umls_knowledge_triples_data(self, frequency_type: str, 
                                           is_1_to_1_triples = True,
                                           is_1_to_N_triples = True,
                                           is_N_to_1_triples = True,
                                           is_N_to_N_triples = True
                                           ):
        umls_knowledge_triples_data = self.__load_umls_knowledge_triples_data()
        
        if frequency_type == None or frequency_type == '' or frequency_type == "all":
            return umls_knowledge_triples_data
        
        min = max = None
        min, max = self.frequency_dict[frequency_type]
        
        umls_knowledge_triples_data = self.__select_triples_indices_by_coexist_num_range(umls_knowledge_triples_data, min_num=min, max_num=max)
        
        print('Loading UMLS Knowledge Triples from UMLSKnowledgeTriplesDataset...')
        # print('############################')
        # print('#        Statistics        #')
        # print('############################')
        print(f"The number of triples is {self.get_n_triples(umls_knowledge_triples_data)}")
        print(f"The number of relations is {self.get_n_relations(umls_knowledge_triples_data)}")
        
        return umls_knowledge_triples_data
        
        
    def __select_triples_indices_by_coexist_num_range(self, umls_knowledge_triples_data, min_num, max_num):
        if min_num is None:
            min_num = 0  # 如果min_num未设置，则默认为0
        if max_num is None:
            max_num = float('inf')  # 如果max_num未设置，则设为无穷大
        if min_num > max_num:
            temp = min_num
            min_num = max_num
            max_num = temp

        selected_umls_knowledge_triples_data = [single_umls_knowledge_triple_data for single_umls_knowledge_triple_data in umls_knowledge_triples_data if min_num <= single_umls_knowledge_triple_data["coexist_num"] < max_num]
        
        return selected_umls_knowledge_triples_data
    
    def get_n_triples(self, umls_knowledge_triples_data):
        return len(umls_knowledge_triples_data)
    
    def get_n_relations(self, umls_knowledge_triples_data):
        relations_set = set()
        for single_data in umls_knowledge_triples_data:
            relation_name = single_data['triple'][1]
            relations_set.add(relation_name)
        return len(relations_set)
    
    def get_knowledge_triples_data_based_on_relationship_type(self, umls_knowledge_triples_data, is_sample = False):
        assert is_sample == True or print("\'is_sample\' must be True, or we can't process the relationship type of knowledge triples data.")
        knowledge_triples_1_1 = []
        knowledge_triples_1_N = []
        knowledge_triples_N_1 = []
        knowledge_triples_N_N = []
        if is_sample:
            for knowledge_triple_data in umls_knowledge_triples_data:
                if knowledge_triple_data['relationship_type'] == '1_to_1':
                    knowledge_triples_1_1.append(knowledge_triple_data)
                elif knowledge_triple_data['relationship_type'] == '1_to_N':
                    knowledge_triples_1_N.append(knowledge_triple_data)
                elif knowledge_triple_data['relationship_type'] == 'N_to_1':
                    knowledge_triples_N_1.append(knowledge_triple_data)
                elif knowledge_triple_data['relationship_type'] == 'N_to_N':
                    knowledge_triples_N_N.append(knowledge_triple_data)
        return knowledge_triples_1_1, knowledge_triples_1_N, knowledge_triples_N_1, knowledge_triples_N_N
            
    

# import sys
# from typing import Any

# import torch
# from torch.utils.data import Dataset

# sys.path.append("/nfs")
# sys.path.append("/nfs/general")
# sys.path.append("/nfs/long_tail")

# import os 
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from utils.my_utils import load_json, random_state, save_json
# from dsets.prompt_dataset import SnomedCTPromptDataset
# from dataclasses import dataclass
# from parrot import Parrot

# def gen_default_parrot():
#     random_state(42)
#     parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
#     return parrot

# default_parrot = gen_default_parrot()

class SnomedCTKnowledgeTriplesEditingCollator:
    
    def __init__(self, prompt_mode: int = 0, 
                 is_demonstration: bool=False,
                 is_rephrase: bool=False,
                 is_locality: bool=False,
                 is_portability: bool=False,
                 rephrase_parrot = None):
        # self.tokenizer = tokenizer
        self.prompt_mode = prompt_mode
        self.prompt_dataset = SnomedCTPromptDataset(prompt_mode=prompt_mode)
        self.is_demonstration = is_demonstration
        self.is_rephrase = is_rephrase
        self.is_locality = is_locality
        self.is_portability = is_portability
        self.rephrase_parrot = rephrase_parrot
        
    def __call__(self, batch):
        indexes = []
        prompts = []
        
        subjects = []
        target_news = []
        rephrase_prompts = []
        
        locality_prompts = []
        locality_ans = []
        
        portability_prompts=  []
        portability_ans = []
        
        for data in batch:
            indexes.append(data['index'])
            triple_with_real_name = data['triple']
            if self.is_demonstration:
                prompt = self.prompt_dataset.transform_triple_with_real_name_to_prompt_with_demonstration(triple_with_real_name)
            else:
                prompt = self.prompt_dataset.transform_triple_with_real_name_to_prompt(triple_with_real_name)
            prompts.append(prompt)
            
            label = data['answers']
            if len(label) == 1:
                label = label[0]
            target_news.append(label)
            
            subject = triple_with_real_name[0]
            subjects.append(subject)
            
            if self.is_rephrase:
                # ! todo
                rephrase_prompt = self.prompt_dataset.generate_rephrase_prompt(original_prompt=prompt, triple_with_real_name=triple_with_real_name)
                rephrase_prompts.append(rephrase_prompt)
            
            if self.is_locality:
                # ! todo
                locality_prompt, answer = self.prompt_dataset.generate_locality_prompt_and_answer(triple_with_real_name=triple_with_real_name)
                locality_prompts.append(locality_prompt)
                locality_ans.append(answer)
                
            if self.is_portability:
                # ! todo
                portability_prompt, answer = self.prompt_dataset.generate_portability_prompt_and_answer(triple_with_real_name=triple_with_real_name)
                portability_prompts.append(portability_prompt)
                portability_ans.append(answer)
                
        rephrase_prompts = rephrase_prompts if self.is_rephrase else None
        
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        } if self.is_locality else None
        
        portability_inputs = {
            'one_hop':{
                'prompt': portability_prompts,
                'ground_truth': portability_ans
            },
        } if self.is_portability else None
        
        return {
            'indexes': indexes,
            'prompts': prompts,
            'subjects': subjects,
            'target_news': target_news,
            'ground_truths': target_news,
            'rephrase_prompts': rephrase_prompts,
            'locality_inputs': locality_inputs,
            'portability_inputs': portability_inputs,
            'batch_len': len(batch)
            }


class SnomedCTKnowledgeTriplesProbingCollator:

    def __init__(self, prompt_mode: int = 0, is_demonstration: bool=False):
        # self.tokenizer = tokenizer
        self.prompt_mode = prompt_mode
        self.prompt_dataset = SnomedCTPromptDataset(prompt_mode=prompt_mode)
        self.is_demonstration = is_demonstration

    def __call__(self, batch):
        indexes = []
        prompts = []
        labels = []
        for data in batch:
            """
            {
                "index": 0,
                "triple": (
                    "Pain of jaw",
                    "Finding site",
                    "Jaw"
                ),
                "coexist_num": 94,
                "answers": [
                    "Jaw"
                ]
            }
            """
            index = data['index']
            triple_with_real_name = data['triple']
            # 这里的 label 可能不止一个
            label = data['answers']
            
            if self.is_demonstration:
                prompt = self.prompt_dataset.transform_triple_with_real_name_to_prompt_with_demonstration(triple_with_real_name)
            else:
                prompt = self.prompt_dataset.transform_triple_with_real_name_to_prompt(triple_with_real_name)
            
            indexes.append(index)
            prompts.append(prompt)
            labels.append(label)

        # inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        return indexes, prompts, labels, len(batch)
    


    
# class UMLSKnowledgeTriplesDatasets(Dataset):
#     def __init__(self, 
#                 frequency_type: str,
#                 save_data_path: str = default_save_knowledge_triples_data_path,
#                 ) -> None:
#         self.frequency_type = frequency_type
#         self.save_data_path = save_data_path
#         self.frequency_dict=  {
#             'long_tail': (0,1),
#             'medium': (2,10),
#             'high': (11, 100),
#             'very_high': (101, 1000),
#             'popular': (1001, None)
#         }
#         min = max = None
#         min, max = self.frequency_dict[self.frequency_type]
#         full_umls_kp_data = self.load_umls_kp_data()
#         # from pdb import set_trace; set_trace()
#         self.umls_kp_data = self.select_triples_indices_by_coexist_num_range(full_umls_kp_data, min_num=min, max_num=max)
        
#     def __len__(self):
#         return len(self.umls_kp_data)
    
#     def __getitem__(self, index):
#         return self.umls_kp_data[index]
    
#     def load_umls_kp_data(self):
#         umls_kp_data_path = os.path.join(self.save_data_path, 'umls_kp.json')
#         umls_kp_data = load_json(umls_kp_data_path)
#         return umls_kp_data
    
#     def select_triples_indices_by_coexist_num_range(self, umls_kp_data, min_num, max_num):
#         if min_num is None:
#             min_num = 0  # 如果min_num未设置，则默认为0
#         if max_num is None:
#             max_num = float('inf')  # 如果max_num未设置，则设为无穷大
#         if min_num > max_num:
#             temp = min_num
#             min_num = max_num
#             max_num = temp

#         selected_umls_kp_data = [single_umls_kp_data for single_umls_kp_data in umls_kp_data if min_num <= single_umls_kp_data["coexist_num"] <= max_num]
        
#         return selected_umls_kp_data

    
    