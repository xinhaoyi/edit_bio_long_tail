import pickle
import sys

import numpy as np

from experiments.parse_ret import RetLoader

sys.path.append("/nfs")
sys.path.append("/nfs/long_tail")
sys.path.append("/nfs/general")

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from typing import Any, List

import torch
import re
from torch.utils.data import Dataset
from transformers import BioGptTokenizer, BioGptForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, set_seed

from entity_linking_preprocess.snomedCT_entity_linking import sapbert_entity_linking

from utils.my_utils import save_json
from utils.my_evaluation import exact_math_score_extend
from dsets.corpus_linked_to_kg_dataset import FilterPubtatorLinkedToSnomedCTDataset, PubtatorLinkedToSnomedCTDataset
from dsets.knowledge_triples_dataset import SnomedCTKnowledgeTriplesDataset
import pandas as pd

import random


from config.path_config import PathConfig


root_path = str(PathConfig.BASE_DIR)
save_dir_path = str(PathConfig.KNOWLEDGE_TRIPLES_DATA_DIR)

class UMLSKnowledgeTriplesDataGenerator:
    def __init__(self) -> None:
        self.pubtator_linked_to_snomedCT_dataset = PubtatorLinkedToSnomedCTDataset()
        self.filtered_pubtator_linked_to_snomedCT_dataset = FilterPubtatorLinkedToSnomedCTDataset(self.pubtator_linked_to_snomedCT_dataset, is_filter_process=False)
        
    def generate(self, is_save: bool=False):
        triple2triple_with_real_name = self.pubtator_linked_to_snomedCT_dataset.load_remaining_triple2triple_with_real_name()
        triple2coexist_num = self.filtered_pubtator_linked_to_snomedCT_dataset.load_filtered_triple2coexist_num()
        triples = self.filtered_pubtator_linked_to_snomedCT_dataset.load_filtered_triples()
        data_list = []
        for index, triple in enumerate(triples):
            coexist_num = triple2coexist_num[triple]
            triple_with_real_name = triple2triple_with_real_name[triple]
            answers = []
            tail_entity_name = triple_with_real_name[2]
            # 移除括号及其中的内容
            filtered_tail_entity_name = re.sub(r'\(.*?\)', '', tail_entity_name)
            # 移除前后的空格和引号
            filtered_tail_entity_name = filtered_tail_entity_name.strip().strip('\'"')
            
            answers.append(filtered_tail_entity_name)
            # answers.append(tail_entity_name)
            # if filtered_tail_entity_name != tail_entity_name:
            #     answers.append(filtered_tail_entity_name)
            
            data = {
                'index': index,
                'triple': triple_with_real_name,
                'coexist_num': coexist_num,
                'answers': answers
            }
            
            data_list.append(data)
            
        if is_save:
            save_json(data=data_list, path='umls_knowledge_triples.json', use_indent=True)
            save_json(data=data_list[:10], path='umls_knowledge_triples_example.json', use_indent=True)
        else:
            return data_list
        

class UMLSKnowledgeTriplesDataFilter:
    def __init__(self, snomedCT_knowledge_triples_dataset=None) -> None:
        if snomedCT_knowledge_triples_dataset == None:
            self.snomedCT_knowledge_triples_dataset = SnomedCTKnowledgeTriplesDataset(frequency_type='popular')
        else:
            self.snomedCT_knowledge_triples_dataset = snomedCT_knowledge_triples_dataset
    
    def get_relations_set_from_triples_data(self, knowledge_triples_data):
        relations_set = set()
        for single_triple_data in knowledge_triples_data:
            relation_name = single_triple_data['triple'][1]
            relations_set.add(relation_name)
        return relations_set
    
    def get_filtered_knowledge_triples_data_on_allowed_relations(self, original_knowledge_triples_data, allowed_relations_set):
        filtered_knowledge_triples_data = []
        for single_triple_data in original_knowledge_triples_data:
            # from pdb import set_trace; set_trace()
            relation_name = single_triple_data['triple'][1]
            if relation_name in allowed_relations_set:
                filtered_knowledge_triples_data.append(single_triple_data)
        
        return filtered_knowledge_triples_data
    
    def parse_data(self, knowledge_triples_data, is_set_N_N_relationship_type = False):
        relation2knowledge_triples_data = {}
        relation2count_knowledge_triples_data = {}
        relation_set = set()
        relations = []
        for single_triple_data in knowledge_triples_data:
            relation_name = single_triple_data['triple'][1]
            relation2knowledge_triples_data.setdefault(relation_name, []).append(single_triple_data)
            relation_set.add(relation_name)
        for relation_name, knowledge_triples_data in relation2knowledge_triples_data.items():
            relation2count_knowledge_triples_data[relation_name] = len(knowledge_triples_data)
        
        relations = list(relation_set)
        
        relation2knowledge_triples_data_1_to_1 = {}
        relation2knowledge_triples_data_1_to_N = {}
        relation2knowledge_triples_data_N_to_1 = {}
        relation2knowledge_triples_data_N_to_N = {}
        
        head_entities = set()
        tail_entities = set()
        
        for relation, knowledge_triples_data in relation2knowledge_triples_data.items():
            head_entity2count = {}
            tail_entity2_count = {}
            for knowledge_triple_data in knowledge_triples_data:
                head_entity = knowledge_triple_data['triple'][0]
                tail_entity = knowledge_triple_data['triple'][2]
                head_entity2count[head_entity] = head_entity2count.get(head_entity, 0) + 1
                tail_entity2_count[tail_entity] = tail_entity2_count.get(tail_entity, 0) + 1
            for single_knowledge_triple_data in knowledge_triples_data:
                    head_entity = single_knowledge_triple_data['triple'][0]
                    tail_entity = single_knowledge_triple_data['triple'][2]
                    head_entities.add(head_entity)
                    tail_entities.add(tail_entity)
            
            if not is_set_N_N_relationship_type:
                for single_knowledge_triple_data in knowledge_triples_data:
                    head_entity = single_knowledge_triple_data['triple'][0]
                    tail_entity = single_knowledge_triple_data['triple'][2]
                    
                    if head_entity2count[head_entity] == 1 and tail_entity2_count[tail_entity] == 1:
                        relation2knowledge_triples_data_1_to_1.setdefault(relation, []).append(single_knowledge_triple_data)
                    elif head_entity2count[head_entity] > 1 and tail_entity2_count[tail_entity] == 1:
                        relation2knowledge_triples_data_N_to_1.setdefault(relation, []).append(single_knowledge_triple_data)
                    elif head_entity2count[head_entity] == 1 and tail_entity2_count[tail_entity] > 1:
                        relation2knowledge_triples_data_1_to_N.setdefault(relation, []).append(single_knowledge_triple_data)
                    else:
                        # N : N
                        relation2knowledge_triples_data_N_to_N.setdefault(relation, []).append(single_knowledge_triple_data)
            else:
                # is_set_N_N_relation_type == True, means the data is sampled data
                for single_knowledge_triple_data in knowledge_triples_data:
                    if single_knowledge_triple_data['relationship_type'] == '1_to_1':
                        relation2knowledge_triples_data_1_to_1.setdefault(relation, []).append(single_knowledge_triple_data)
                    elif single_knowledge_triple_data['relationship_type'] == '1_to_N':
                        relation2knowledge_triples_data_1_to_N.setdefault(relation, []).append(single_knowledge_triple_data)
                    elif single_knowledge_triple_data['relationship_type'] == 'N_to_1':
                        relation2knowledge_triples_data_N_to_1.setdefault(relation, []).append(single_knowledge_triple_data)
                    elif single_knowledge_triple_data['relationship_type'] == 'N_to_N':
                        # N : N
                        relation2knowledge_triples_data_N_to_N.setdefault(relation, []).append(single_knowledge_triple_data)
        
        # from pdb import set_trace; set_trace()
        relation2count_knowledge_triples_data_1_to_1 = {}
        for relation_name, knowledge_triples_data in relation2knowledge_triples_data_1_to_1.items():
            relation2count_knowledge_triples_data_1_to_1[relation_name] = len(knowledge_triples_data)
        
        # from pdb import set_trace; set_trace()
        
        relation2count_knowledge_triples_data_1_to_N = {}
        for relation_name, knowledge_triples_data in relation2knowledge_triples_data_1_to_N.items():
            relation2count_knowledge_triples_data_1_to_N[relation_name] = len(knowledge_triples_data)
            
        relation2count_knowledge_triples_data_N_to_1 = {}
        for relation_name, knowledge_triples_data in relation2knowledge_triples_data_N_to_1.items():
            relation2count_knowledge_triples_data_N_to_1[relation_name] = len(knowledge_triples_data)
        
        relation2count_knowledge_triples_data_N_to_N = {}
        for relation_name, knowledge_triples_data in relation2knowledge_triples_data_N_to_N.items():
            relation2count_knowledge_triples_data_N_to_N[relation_name] = len(knowledge_triples_data)
        
        overall_count_knowledge_triples_data = sum([len(knowledge_triples_data) for knowledge_triples_data in relation2knowledge_triples_data.values()])
        overall_count_knowledge_triples_data_1_to_1 = sum([len(knowledge_triples_data) for knowledge_triples_data in relation2knowledge_triples_data_1_to_1.values()])
        overall_count_knowledge_triples_data_1_to_N = sum([len(knowledge_triples_data) for knowledge_triples_data in relation2knowledge_triples_data_1_to_N.values()])
        overall_count_knowledge_triples_data_N_to_1 = sum([len(knowledge_triples_data) for knowledge_triples_data in relation2knowledge_triples_data_N_to_1.values()])
        overall_count_knowledge_triples_data_N_to_N = sum([len(knowledge_triples_data) for knowledge_triples_data in relation2knowledge_triples_data_N_to_N.values()])
        
        # from pdb import set_trace; set_trace()
                
        return {
            'relations': relations,
            'head_entities': list(head_entities),
            'tail_entities': list(tail_entities),
            'relation2knowledge_triples_data': relation2knowledge_triples_data,
            'relation2count_knowledge_triples_data': relation2count_knowledge_triples_data,
            'relation2knowledge_triples_data_1_to_1': relation2knowledge_triples_data_1_to_1,
            'relation2knowledge_triples_data_1_to_N': relation2knowledge_triples_data_1_to_N,
            'relation2knowledge_triples_data_N_to_1': relation2knowledge_triples_data_N_to_1,
            'relation2knowledge_triples_data_N_to_N': relation2knowledge_triples_data_N_to_N,
            'relation2count_knowledge_triples_data_1_to_1': relation2count_knowledge_triples_data_1_to_1,
            'relation2count_knowledge_triples_data_1_to_N': relation2count_knowledge_triples_data_1_to_N,
            'relation2count_knowledge_triples_data_N_to_1': relation2count_knowledge_triples_data_N_to_1,
            'relation2count_knowledge_triples_data_N_to_N': relation2count_knowledge_triples_data_N_to_N,
            'overall_count_knowledge_triples_data': overall_count_knowledge_triples_data,
            'overall_count_knowledge_triples_data_1_to_1': overall_count_knowledge_triples_data_1_to_1,
            'overall_count_knowledge_triples_data_1_to_N': overall_count_knowledge_triples_data_1_to_N,
            'overall_count_knowledge_triples_data_N_to_1': overall_count_knowledge_triples_data_N_to_1,
            'overall_count_knowledge_triples_data_N_to_N': overall_count_knowledge_triples_data_N_to_N
        }
        
    def gen_df_data(self, relation2count, relation2count_1_to_1, relation2count_1_to_N, relation2count_N_to_1, relation2count_N_to_N):
        data = []
        count_total_triples_in_distribution = sum(relation2count.values())
        
        count_total_1_1 = sum(relation2count_1_to_1.values())
        proportion_total_1_to_1 = f"{count_total_1_1 / count_total_triples_in_distribution * 100:.2f}%"
        
        count_total_1_N = sum(relation2count_1_to_N.values())
        proportion_total_1_to_N = f"{count_total_1_N / count_total_triples_in_distribution * 100:.2f}%"
        
        count_total_N_1 = sum(relation2count_N_to_1.values())
        proportion_total_N_to_1 = f"{count_total_N_1 / count_total_triples_in_distribution * 100:.2f}%"
        
        count_total_N_N = sum(relation2count_N_to_N.values())
        proportion_total_N_to_N = f"{count_total_N_N / count_total_triples_in_distribution * 100:.2f}%"
        
        for relation in relation2count.keys():
            count = relation2count.get(relation, 0)
            proportion = f"{count / count_total_triples_in_distribution * 100:.2f}%"
            
            count_1_to_1 = relation2count_1_to_1.get(relation, 0)
            proportion_1_to_1 = f"{count_1_to_1 / relation2count[relation] * 100:.2f}%"
            
            count_1_to_N = relation2count_1_to_N.get(relation, 0)
            proportion_1_to_N = f"{count_1_to_N / relation2count[relation] * 100:.2f}%"
            
            count_N_to_1 = relation2count_N_to_1.get(relation, 0)
            proportion_N_to_1 = f"{count_N_to_1 / relation2count[relation] * 100:.2f}%"
            
            count_N_to_N = relation2count_N_to_N.get(relation, 0)
            proportion_N_to_N = f"{count_N_to_N / relation2count[relation] * 100:.2f}%"
            
            data.append([relation,f"({count}/{count_total_triples_in_distribution}) {proportion}", f"{count_1_to_1} ({proportion_1_to_1})", f"{count_1_to_N} ({proportion_1_to_N})", f"{count_N_to_1} ({proportion_N_to_1})", f"{count_N_to_N} ({proportion_N_to_N})"])
        
        data.append(['[ALL_RELATIONS]',
                     f"{count_total_triples_in_distribution} ({count_total_triples_in_distribution})",
                     f"{count_total_1_1} ({proportion_total_1_to_1})",
                     f"{count_total_1_N} ({proportion_total_1_to_N})",
                     f"{count_total_N_1} ({proportion_total_N_to_1})",
                     f"{count_total_N_N} ({proportion_total_N_to_N})"])
        
        df = pd.DataFrame(data, columns=['Relation', 'total', '1_1', '1_N', 'N_1', 'N_N'])
        
        return df
        
    def analyse_one_distribution_data(self, distribution: str, is_set_N_N_relationship_type: bool=False):
        """
        distribution: 'popular', 'very_high', 'high', 'medium', 'long_tail'
        """
        distribution_name = distribution.replace('_', ' ')
        # original popular
        
        # from pdb import set_trace; set_trace()
        
        original_knowledge_triples_data = self.snomedCT_knowledge_triples_dataset.get_umls_knowledge_triples_data(distribution)
        relations_set = self.get_relations_set_from_triples_data(knowledge_triples_data=original_knowledge_triples_data)
        
        print("\n\n")
        print("###################################################")
        print(f"#       For {distribution.capitalize()} Knowledge Triples         #")
        print("###################################################")
        
        
        knowledge_triples_data_parsed_dict = self.parse_data(original_knowledge_triples_data, is_set_N_N_relationship_type=is_set_N_N_relationship_type)
        
        print(f"The total num of head entities is {len(knowledge_triples_data_parsed_dict['head_entities'])}")
        print(f"The total num of tail entities is {len(knowledge_triples_data_parsed_dict['tail_entities'])}")
        
        print(f"The relations and their num of triples:\nThe total number of triples is {knowledge_triples_data_parsed_dict['overall_count_knowledge_triples_data']}\n", knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data'],'\n')
        print(f"The relations and their num of 1:1 triples:\nThe total number of 1:1 triples is {knowledge_triples_data_parsed_dict['overall_count_knowledge_triples_data_1_to_1']}\n", knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_1_to_1'],'\n')
        print(f"The relations and their num of 1:N triples:\nThe total number of 1:N triples is {knowledge_triples_data_parsed_dict['overall_count_knowledge_triples_data_1_to_N']}\n", knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_1_to_N'],'\n')
        print(f"The relations and their num of N:1 triples:\nThe total number of N:1 triples is {knowledge_triples_data_parsed_dict['overall_count_knowledge_triples_data_N_to_1']}\n", knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_N_to_1'],'\n')
        print(f"The relations and their num of N:N triples:\nThe total number of N:N triples is {knowledge_triples_data_parsed_dict['overall_count_knowledge_triples_data_N_to_N']}\n", knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_N_to_N'],'\n')
        
        df = self.gen_df_data(relation2count=knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data'],
                        relation2count_1_to_1=knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_1_to_1'],
                        relation2count_1_to_N=knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_1_to_N'],
                        relation2count_N_to_1=knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_N_to_1'],
                        relation2count_N_to_N=knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_N_to_N']
                        )
                
        print(f"{distribution} Knowledge Triples Statistics Table")
        print(df)
        print('\n')
        
        
    def analyse_all_data(self, is_sampled_data: bool = False):
        """
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
        frequency_dict = self.snomedCT_knowledge_triples_dataset.frequency_dict
        distribution_list = ['all'] + list(frequency_dict.keys())
        # distribution_list = list(frequency_dict.keys())
        
        for distribution in distribution_list:
            self.analyse_one_distribution_data(distribution=distribution, is_set_N_N_relationship_type=is_sampled_data)
    
    
    def filter_data(self, knowledge_triples_data, is_filter_1_to_1: bool, is_filter_1_to_N: bool, is_filter_N_to_1: bool, is_filter_N_to_N: bool, relations_for_removal: list, is_save: bool=False):
        # knowledge_triples_data = self.snomedCT_knowledge_triples_dataset.get_umls_knowledge_triples_data(frequency_type='')
        
        relation_set = self.get_relations_set_from_triples_data(knowledge_triples_data)
        
        relation_set.difference_update(relations_for_removal)
        
        knowledge_triples_data_parsed_dict = self.parse_data(knowledge_triples_data)
        
        filtered_knowledge_triples_data = []
        # from pdb import set_trace; set_trace()
        if is_filter_1_to_1 is False:
            relation2knowledge_triples_data_1_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_1']
            for knowledge_triples_data_1_to_1 in relation2knowledge_triples_data_1_to_1.values():
                filtered_knowledge_triples_data.extend(knowledge_triples_data_1_to_1)
        
        if is_filter_1_to_N is False:
            relation2knowledge_triples_data_1_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_N']
            for knowledge_triples_data_1_to_N in relation2knowledge_triples_data_1_to_N.values():
                filtered_knowledge_triples_data.extend(list(knowledge_triples_data_1_to_N))
        
        if is_filter_N_to_1 is False:
            relation2knowledge_triples_data_N_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_1']
            for knowledge_triples_data_N_to_1 in relation2knowledge_triples_data_N_to_1.values():
                filtered_knowledge_triples_data.extend(list(knowledge_triples_data_N_to_1))
        
        if is_filter_N_to_N is False:
            relation2knowledge_triples_data_N_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_N']
            for knowledge_triples_data_N_to_N in relation2knowledge_triples_data_N_to_N.values():
                filtered_knowledge_triples_data.extend(list(knowledge_triples_data_N_to_N))
        
        # from pdb import set_trace; set_trace()
        filtered_knowledge_triples_data = self.get_filtered_knowledge_triples_data_on_allowed_relations(original_knowledge_triples_data=filtered_knowledge_triples_data, allowed_relations_set=relation_set)
        
        if is_save:
            save_json(filtered_knowledge_triples_data, 'umls_knowledge_triples_filtered.json', use_indent=True)
        else: 
            return filtered_knowledge_triples_data
        
        # return filtered_knowledge_triples_data
                
        
    def update_NN_relationship_type_to_original_data(self, knowledge_triples_data, is_save: bool=False):
        knowledge_triples_data_parsed_dict = self.parse_data(knowledge_triples_data)
        relation2knowledge_triples_data_1_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_1']
        relation2knowledge_triples_data_1_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_N']
        relation2knowledge_triples_data_N_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_1']
        relation2knowledge_triples_data_N_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_N']
        
        all_updated_knowledge_triples = []
        
        for _, knowledge_triples_data_1_to_1 in relation2knowledge_triples_data_1_to_1.items():
            for single_knowledge_triple_data_1_to_1 in knowledge_triples_data_1_to_1:
                single_knowledge_triple_data_1_to_1.update({"relationship_type": "1_to_1"})
                all_updated_knowledge_triples.append(single_knowledge_triple_data_1_to_1)
        for _, knowledge_triples_data_1_to_N in relation2knowledge_triples_data_1_to_N.items():
            for single_knowledge_triple_data_1_to_N in knowledge_triples_data_1_to_N:
                single_knowledge_triple_data_1_to_N.update({"relationship_type": "1_to_N"})
                all_updated_knowledge_triples.append(single_knowledge_triple_data_1_to_N)
                
        for _, knowledge_triples_data_N_to_1 in relation2knowledge_triples_data_N_to_1.items():
            for single_knowledge_triple_data_N_to_1 in knowledge_triples_data_N_to_1:
                single_knowledge_triple_data_N_to_1.update({"relationship_type": "N_to_1"})
                all_updated_knowledge_triples.append(single_knowledge_triple_data_N_to_1)
                
        for _, knowledge_triples_data_N_to_N in relation2knowledge_triples_data_N_to_N.items():
            for single_knowledge_triple_data_N_to_N in knowledge_triples_data_N_to_N:
                single_knowledge_triple_data_N_to_N.update({"relationship_type": "N_to_N"})
                all_updated_knowledge_triples.append(single_knowledge_triple_data_N_to_N)
        
        all_updated_knowledge_triples = sorted(all_updated_knowledge_triples, key=lambda x: x['index'])
        
        if is_save:
            save_json(data=all_updated_knowledge_triples, path='clinic_knowledge_triples.json', use_indent=True)
            save_json(data=all_updated_knowledge_triples[:10], path='clinic_knowledge_triples_example.json', use_indent=True)
        else:
            return all_updated_knowledge_triples
        
    def update_locality_data_to_original_data(self, knowledge_triples_data, is_save: bool=False):
        knowledge_triples_data_parsed_dict = self.parse_data(knowledge_triples_data)
        relation2knowledge_triples_data = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data']
        # relation2knowledge_triples_data_1_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_1']
        # relation2knowledge_triples_data_1_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_N']
        # relation2knowledge_triples_data_N_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_1']
        # relation2knowledge_triples_data_N_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_N']
        
        # single_knowledge_triple_data_1_to_N.update({"relationship_type": "1_to_N"})
        random.seed(44)
        
        for single_knowledge_triple_data in knowledge_triples_data:
            single_knowledge_triple = single_knowledge_triple_data["triple"]
            relation = single_knowledge_triple[1]
            knowledge_triples_data_list = relation2knowledge_triples_data[relation]
            
            while(True):
                selected_single_knowledge_triple_data = random.choice(knowledge_triples_data_list)
                selected_single_knowledge_triple = selected_single_knowledge_triple_data["triple"]
            
                if single_knowledge_triple != selected_single_knowledge_triple:
                    break
            
            single_knowledge_triple_data.update({"locality_triple": selected_single_knowledge_triple})
        
        if is_save:
            save_json(data=knowledge_triples_data, path='knowledge_triples_with_locality.json', use_indent=True)
            save_json(data=knowledge_triples_data[:10], path='clinic_knowledge_triples_with_locality_example.json', use_indent=True)
        else:
            return knowledge_triples_data
        
    
    def filter_zero_coexist_num_data(self, knowledge_triples_data, is_save: bool=False):
        filtered_knowledge_triples_data = []
        
        for single_knowledge_triple_data in knowledge_triples_data:
            if single_knowledge_triple_data['coexist_num'] != 0:
                filtered_knowledge_triples_data.append(single_knowledge_triple_data)
        
        if is_save:
            save_json(data=filtered_knowledge_triples_data, path='knowledge_triples_zero_coexist_num.json', use_indent=True)
            save_json(data=filtered_knowledge_triples_data[:10], path='clinic_knowledge_triples_zero_coexist_num_example.json', use_indent=True)
        else:
            return filtered_knowledge_triples_data
        
    def rebuild_data_index(self, knowledge_triples_data, is_save: bool=False):
        
        filtered_knowledge_triples_data = []
        
        for new_idx, single_knowledge_triple_data in enumerate(knowledge_triples_data):
            single_knowledge_triple_data["index"] = new_idx
            filtered_knowledge_triples_data.append(single_knowledge_triple_data)
        
        if is_save:
            save_json(data=filtered_knowledge_triples_data, path='knowledge_triples_zero_coexist_num.json', use_indent=True)
            save_json(data=filtered_knowledge_triples_data[:10], path='clinic_knowledge_triples_zero_coexist_num_example.json', use_indent=True)
        else:
            return filtered_knowledge_triples_data
        
        
            
            
            
            
        
        
        
        
    def dc_sample_data(self, knowledge_triples_data, total_sample_size = 1250):
        """
        First of all, I want to calculate how many triples for each relation, multiply the ratio of the number of triples corresponding to this relation to the total number of triples in the original data by the number of samples I want to take to get N.
        Check if the number of N is less than the number of triples corresponding to this relation, if N is particularly small, for example, less than (total number of samples/number of relations) * 0.1, we make up this min(num_relations, (total number of samples/number of relations) * 0.5)
        After this is done, I need to figure out if the sum exceeds my sampling value, and if it does, reduce the number of samples for each relation proportionally.
        Next, for each relation, I have to make sure that 1:1, ..... N:N
        
        首先,我要算每个relation取多少个triples, 用原始数据中这个relation对应triples数量占总triples数量的比例 乘以我要采样的个数, 得到N
        检查N数量是不是比这个relation对应的triples数量要少, 如果N特别少, 比如少于 (总采样数/relation数量) * 0.1, 我们就补成这 min(num_relations, (总采样数/relation数量) * 0.5)
        这样结束了之后, 我要算一下是不是总和超过我的采样值了， 如果总和超过了采样值, 按比例减少每个relation的采样数量
        接下来, 对于每个relation, 我都要确保1:1, ..... N:N
        
        """
        random.seed(42)
        
        samples = []
        
        knowledge_triples_data_parsed_dict = self.parse_data(knowledge_triples_data)
        
        relations = knowledge_triples_data_parsed_dict['relations']
        count_relations = len(relations)
        
        relation2sample_allocation = {}
        total_count_knowledge_triples_data = knowledge_triples_data_parsed_dict['overall_count_knowledge_triples_data']
        
        if total_count_knowledge_triples_data <= total_sample_size:
            
            # for single_data in knowledge_triples_data:
            #     # todo 
            #     single_data.updata({'relationship_type': })
            
            # sampled_knowledge_triples_data = knowledge_triples_data
            # return sampled_knowledge_triples_data
            total_sample_size = total_count_knowledge_triples_data
            return knowledge_triples_data
        
        relation2count_knowledge_triples_data = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data']
        
        print("********** Decide allocation for each relation **********")
        
        for index, (relation_name, count_knowledge_triples_data) in enumerate(relation2count_knowledge_triples_data.items()):
            print(f"\nProcesisng '{relation_name}' ({index} / {len(relations) - 1})...")
            proportion = count_knowledge_triples_data / total_count_knowledge_triples_data
            N = round(proportion * total_sample_size)
            
            # Check and change N:
            if N < (total_sample_size / count_relations) * 0.05:
                N = min(count_knowledge_triples_data, round((total_sample_size / count_relations) * 0.05))
            relation2sample_allocation[relation_name] = N
        
        # 如果总和超过了采样值，首先按比例减少每个relation的采样数量
        if sum(relation2sample_allocation.values()) > total_sample_size:
            print(f"The num of current samples is {sum(relation2sample_allocation.values())}, larger than total sample size {total_sample_size}, we firstly decrease the data following sliding scale.")
            adjustment_ratio = total_sample_size / sum(relation2sample_allocation.values())
            relation2sample_allocation = {relation: int(count * adjustment_ratio) for relation, count in relation2sample_allocation.items()}
        
        # 然后继续调整，让每个relation采样的和加起来 恰好等于 总采样量
        random.shuffle(relations)  # 随机打乱relation顺序
        current_total_count_samples = sum(relation2sample_allocation.values())
        difference = total_sample_size - current_total_count_samples
        
        print(f"After the first decrease, the num of current samples is now {current_total_count_samples}, while the total sample size is {total_sample_size}, the difference is {difference}")
        
        # 如果总数小于目标采样量，随机增加某些relation的采样数
        while difference > 0:
            for relation in relations:
                if difference <= 0:
                    break
                # 仅在不超过该relation原始数量的前提下增加
                if relation2sample_allocation[relation] < relation2count_knowledge_triples_data[relation]:
                    relation2sample_allocation[relation] += 1
                    difference -= 1
                # random.shuffle(relations)  # 再次随机打乱以保持随机性

        # 如果总数大于目标采样量，随机减少某些relation的采样数
        while difference < 0:
            for relation in relations:
                if difference >= 0:
                    break
                # 保证每个relation至少有min_sample_size个样本
                min_sample_size = min(count_knowledge_triples_data, round((total_sample_size / count_relations) * 0.5))
                if relation2sample_allocation[relation] > min_sample_size:
                    relation2sample_allocation[relation] -= 1
                    difference += 1
                # random.shuffle(relations)  # 再次随机打乱以保持随机性
        
        print(f"After second decrease, the total the num of current samples is now {sum(relation2sample_allocation.values())}, while the total sample size is {total_sample_size}, the difference is {difference}")
        
        # 接下来，我希望尽可能保留每个relation对应的triples的1:1, 1:N, N:1, N:N的关系比例
        print("*************Next, we'll calculte the count_samples_1_to_1, count_samples_1_to_N, count_samples_N_to_1, count_samples_N_to_N for each relation.")
        relation2count_knowledge_triples_data_1_to_1 = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_1_to_1']
        relation2count_knowledge_triples_data_1_to_N = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_1_to_N']
        relation2count_knowledge_triples_data_N_to_1 = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_N_to_1']
        relation2count_knowledge_triples_data_N_to_N = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_N_to_N']
        
        relation2knowledge_triples_data_1_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_1']
        relation2knowledge_triples_data_1_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_N']
        relation2knowledge_triples_data_N_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_1']
        relation2knowledge_triples_data_N_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_N']
                
        for index, relation in enumerate(relation2count_knowledge_triples_data.keys()):
            print(f"\nProcesisng '{relation}' ({index} / {len(relations) - 1})...")
            count_knowledge_triples_data_1_to_1 = relation2count_knowledge_triples_data_1_to_1[relation] if relation in relation2count_knowledge_triples_data_1_to_1.keys() else 0
            count_knowledge_triples_data_1_to_N = relation2count_knowledge_triples_data_1_to_N[relation] if relation in relation2count_knowledge_triples_data_1_to_N.keys() else 0
            count_knowledge_triples_data_N_to_1 = relation2count_knowledge_triples_data_N_to_1[relation] if relation in relation2count_knowledge_triples_data_N_to_1.keys() else 0
            count_knowledge_triples_data_N_to_N = relation2count_knowledge_triples_data_N_to_N[relation] if relation in relation2count_knowledge_triples_data_N_to_N.keys() else 0
            
            sample_allocation = relation2sample_allocation[relation]
            
            total_count_knowledge_triples_data = count_knowledge_triples_data_1_to_1 + count_knowledge_triples_data_1_to_N + count_knowledge_triples_data_N_to_1 + count_knowledge_triples_data_N_to_N
            
            knowledge_triples_data_1_to_1 = relation2knowledge_triples_data_1_to_1[relation] if relation in relation2knowledge_triples_data_1_to_1.keys() else []
            knowledge_triples_data_1_to_N = relation2knowledge_triples_data_1_to_N[relation] if relation in relation2knowledge_triples_data_1_to_N.keys() else []
            knowledge_triples_data_N_to_1 = relation2knowledge_triples_data_N_to_1[relation] if relation in relation2knowledge_triples_data_N_to_1.keys() else []
            knowledge_triples_data_N_to_N = relation2knowledge_triples_data_N_to_N[relation] if relation in relation2knowledge_triples_data_N_to_N.keys() else []
            
            proportion_1_to_1 = count_knowledge_triples_data_1_to_1 / total_count_knowledge_triples_data
            proportion_1_to_N = count_knowledge_triples_data_1_to_N / total_count_knowledge_triples_data
            proportion_N_to_1 = count_knowledge_triples_data_N_to_1 / total_count_knowledge_triples_data
            proportion_N_to_N = count_knowledge_triples_data_N_to_N / total_count_knowledge_triples_data
            
            count_samples_1_to_1 = round(proportion_1_to_1 * sample_allocation)
            count_samples_1_to_N = round(proportion_1_to_N * sample_allocation)
            count_samples_N_to_1 = round(proportion_N_to_1 * sample_allocation)
            count_samples_N_to_N = round(proportion_N_to_N * sample_allocation)
            
            current_count_total_samples_single_relation = 0
            
            current_count_total_samples_single_relation = count_samples_1_to_1 + count_samples_1_to_N + count_samples_N_to_1 + count_samples_N_to_N
            
            difference_single_relation =  sample_allocation - current_count_total_samples_single_relation
            
            print(f"Now, the sample allocation for '{relation}' is {sample_allocation}, while the sum of samples_1_to_1 ..to.. samples_N_to_N is {current_count_total_samples_single_relation}, the difference is {difference_single_relation}")
            print(f"The samples of 1_to_1 is {count_samples_1_to_1}")
            print(f"The samples of 1_to_N is {count_samples_1_to_N}")
            print(f"The samples of N_to_1 is {count_samples_N_to_1}")
            print(f"The samples of N_to_N is {count_samples_N_to_N}")
            
            tempt_categories = ['1_to_1', '1_to_N', 'N_to_1', 'N_to_N']
            
            is_further_filtering = False
            
            while difference_single_relation > 0:
                is_further_filtering = True if is_further_filtering == False else True
                if difference_single_relation <= 0:
                    break
                chosen_category = random.choice(tempt_categories)
                if '1_to_1' == chosen_category and count_samples_1_to_1 > 0:
                    if count_samples_1_to_1 < count_knowledge_triples_data_1_to_1:
                        count_samples_1_to_1 = count_samples_1_to_1 + 1
                        difference_single_relation = difference_single_relation -1
                elif '1_to_N' == chosen_category and count_samples_1_to_N > 0:
                    if count_samples_1_to_N < count_knowledge_triples_data_1_to_N:
                        count_samples_1_to_N = count_samples_1_to_N + 1
                        difference_single_relation = difference_single_relation -1
                elif 'N_to_1' == chosen_category and count_samples_N_to_1 > 0:
                    if count_samples_N_to_1 < count_knowledge_triples_data_N_to_1:
                        count_samples_N_to_1 = count_samples_N_to_1 + 1
                        difference_single_relation = difference_single_relation -1
                elif 'N_to_N' == chosen_category and count_samples_N_to_N > 0:
                    if count_samples_N_to_N < count_knowledge_triples_data_N_to_N:
                        count_samples_N_to_N = count_samples_N_to_N + 1
                        difference_single_relation = difference_single_relation -1
            
            while difference_single_relation < 0:
                is_further_filtering = True if is_further_filtering == False else True
                if difference_single_relation >= 0:
                    break
                chosen_category = random.choice(tempt_categories)
                if '1_to_1' == chosen_category and count_samples_1_to_1 > 0:
                    if count_samples_1_to_1 > 0:
                        count_samples_1_to_1 = count_samples_1_to_1 - 1
                        difference_single_relation = difference_single_relation + 1
                elif '1_to_N' == chosen_category and count_samples_1_to_N > 0:
                    if count_samples_1_to_N > 0:
                        count_samples_1_to_N = count_samples_1_to_N - 1
                        difference_single_relation = difference_single_relation + 1
                elif 'N_to_1' == chosen_category and count_samples_N_to_1 > 0:
                    if count_samples_N_to_1 > 0:
                        count_samples_N_to_1 = count_samples_N_to_1 - 1
                        difference_single_relation = difference_single_relation + 1
                elif 'N_to_N' == chosen_category and count_samples_N_to_N > 0:
                    if count_samples_N_to_N > 0:
                        count_samples_N_to_N = count_samples_N_to_N - 1
                        difference_single_relation = difference_single_relation + 1
            
            if is_further_filtering:
                current_count_total_samples_single_relation = count_samples_1_to_1 + count_samples_1_to_N + count_samples_N_to_1 + count_samples_N_to_N
                print(f"After further processing, now, the sample allocation for '{relation}' is {sample_allocation}, while the sum of samples_1_to_1 ..to.. samples_N_to_N is {current_count_total_samples_single_relation}, the difference is {difference_single_relation}")
                print(f"The samples of 1_to_1 is {count_samples_1_to_1}")
                print(f"The samples of 1_to_N is {count_samples_1_to_N}")
                print(f"The samples of N_to_1 is {count_samples_N_to_1}")
                print(f"The samples of N_to_N is {count_samples_N_to_N}")
            
            print(f"Just before sampling, the 1_to_1 is {count_samples_1_to_1}, the 1_to_N is {count_samples_1_to_N}, the N_to_1 is {count_samples_N_to_1}, the N_to_N is {count_samples_N_to_N}...")
            
            samples_1_to_1_single_relation = []
            samples_1_to_N_single_relation = []
            samples_N_to_1_single_relation = []
            samples_N_to_N_single_relation = []
            
            random.seed(42)
            # from pdb import set_trace; set_trace()
            if count_samples_1_to_1 > 0:
                random.seed(42)
                # samples_1_to_1_single_relation = random.sample(knowledge_triples_data_1_to_1, count_samples_1_to_1)
                samples_1_to_1_single_relation = [
                    {**sample, 'relationship_type': '1_to_1'} for sample in random.sample(knowledge_triples_data_1_to_1, count_samples_1_to_1)
                ]
                
                samples.extend(samples_1_to_1_single_relation)
            
            if count_samples_1_to_N > 0:
                random.seed(42)
                # samples_1_to_N_single_relation = random.sample(knowledge_triples_data_1_to_N, count_samples_1_to_N)
                samples_1_to_N_single_relation = [
                    {**sample, 'relationship_type': '1_to_N'} for sample in random.sample(knowledge_triples_data_1_to_N, count_samples_1_to_N)
                ]
                samples.extend(samples_1_to_N_single_relation)
                
                
            if count_samples_N_to_1 > 0:
                random.seed(42)
                # samples_N_to_1_single_relation = random.sample(knowledge_triples_data_N_to_1, count_samples_N_to_1)
                samples_N_to_1_single_relation = [
                    {**sample, 'relationship_type': 'N_to_1'} for sample in random.sample(knowledge_triples_data_N_to_1, count_samples_N_to_1)
                ]
                samples.extend(samples_N_to_1_single_relation)
                
            if count_samples_N_to_N > 0:
                random.seed(42)
                # samples_N_to_N_single_relation = random.sample(knowledge_triples_data_N_to_N, count_samples_N_to_N)
                samples_N_to_N_single_relation = [
                    {**sample, 'relationship_type': 'N_to_N'} for sample in random.sample(knowledge_triples_data_N_to_N, count_samples_N_to_N)
                ]
                samples.extend(samples_N_to_N_single_relation)
            
            current_count_total_samples_single_relation = len(samples_1_to_1_single_relation) + len(samples_1_to_N_single_relation) + len(samples_N_to_1_single_relation) + len(samples_N_to_N_single_relation)
            
            print(f"Finally, In this relation {relation}. After the actual sampling, the total triples for this relation is {current_count_total_samples_single_relation}")
            
            print(f"The samples_1_to_1 is {len(samples_1_to_1_single_relation)}")
            print(f"The samples_1_to_N is {len(samples_1_to_N_single_relation)}")
            print(f"The samples_N_to_1 is {len(samples_N_to_1_single_relation)}")
            print(f"The samples_N_to_N is {len(samples_N_to_N_single_relation)}")
            
            print(f"The current total num of samples is {len(samples)}")
            
        
        samples_sorted = sorted(samples, key=lambda x: x['index'])
        
        # Step 2: Refactor index to start at 0 and increment it.
        # for i, sample in enumerate(samples_sorted):
        #     sample['index'] = i
        
        return samples_sorted
    
    
    
    def eq_sample_data(self, knowledge_triples_data, total_sample_size = 2000):
        random.seed(42)
        samples = []
        
        knowledge_triples_data_parsed_dict = self.parse_data(knowledge_triples_data)
        
        relations = knowledge_triples_data_parsed_dict['relations']
        count_relations = len(relations)
        
        relation2sample_allocation = {}
        total_count_knowledge_triples_data = knowledge_triples_data_parsed_dict['overall_count_knowledge_triples_data']
        
        if total_count_knowledge_triples_data <= total_sample_size:
            
            # for single_data in knowledge_triples_data:
            #     # todo 
            #     single_data.updata({'relationship_type': })
            
            # sampled_knowledge_triples_data = knowledge_triples_data
            # return sampled_knowledge_triples_data
            total_sample_size = total_count_knowledge_triples_data
            
        
        # overall_count_knowledge_triples_data = knowledge_triples_data_parsed_dict['overall_count_knowledge_triples_data']
        
        
        # 'relations': relations,
        # 'relation2knowledge_triples_data': relation2knowledge_triples_data,
        # 'relation2count_knowledge_triples_data': relation2count_knowledge_triples_data,
        # 'relation2knowledge_triples_data_1_to_1': relation2knowledge_triples_data_1_to_1,
        # 'relation2knowledge_triples_data_1_to_N': relation2knowledge_triples_data_1_to_N,
        # 'relation2knowledge_triples_data_N_to_1': relation2knowledge_triples_data_N_to_1,
        # 'relation2knowledge_triples_data_N_to_N': relation2knowledge_triples_data_N_to_N,
        # 'relation2count_knowledge_triples_data_1_to_1': relation2count_knowledge_triples_data_1_to_1,
        # 'relation2count_knowledge_triples_data_1_to_N': relation2count_knowledge_triples_data_1_to_N,
        # 'relation2count_knowledge_triples_data_N_to_1': relation2count_knowledge_triples_data_N_to_1,
        # 'relation2count_knowledge_triples_data_N_to_N': relation2count_knowledge_triples_data_N_to_N,
        # 'overall_count_knowledge_triples_data': overall_count_knowledge_triples_data,
        # 'overall_count_knowledge_triples_data_1_to_1': overall_count_knowledge_triples_data_1_to_1,
        # 'overall_count_knowledge_triples_data_1_to_N': overall_count_knowledge_triples_data_1_to_N,
        # 'overall_count_knowledge_triples_data_N_to_1': overall_count_knowledge_triples_data_N_to_1,
        # 'overall_count_knowledge_triples_data_N_to_N': overall_count_knowledge_triples_data_N_to_N
        
        
        all_count_knowledge_triples_data = [knowledge_triples_data_parsed_dict[key] for key in ['overall_count_knowledge_triples_data_1_to_1', 'overall_count_knowledge_triples_data_1_to_N', 'overall_count_knowledge_triples_data_N_to_1', 'overall_count_knowledge_triples_data_N_to_N']]
        
        count_knowledge_triples_data_1_1, count_knowledge_triples_data_1_N, count_knowledge_triples_data_N_1, count_knowledge_triples_data_N_N = all_count_knowledge_triples_data
        
        # Decide allocation for total 1:1, 1:N, N:1, N:N
        print("********** Decide allocation for 1:1, 1:N, N:1, N:N **********")
        
        avg_allocation = int(total_sample_size / 4)
        
        num_group = 4
        
        allocation_knowledge_triples_data_1_1 = avg_allocation if avg_allocation < count_knowledge_triples_data_1_1 else count_knowledge_triples_data_1_1
        allocation_knowledge_triples_data_1_N = avg_allocation if avg_allocation < count_knowledge_triples_data_1_N else count_knowledge_triples_data_1_N
        allocation_knowledge_triples_data_N_1 = avg_allocation if avg_allocation < count_knowledge_triples_data_N_1 else count_knowledge_triples_data_N_1
        allocation_knowledge_triples_data_N_N = avg_allocation if avg_allocation < count_knowledge_triples_data_N_N else count_knowledge_triples_data_N_N
        
        print(f"After the first round, the 1_to_1 allocation is {allocation_knowledge_triples_data_1_1}, the 1_to_N allocation is {allocation_knowledge_triples_data_1_N}, the N_to_1 allocation is {count_knowledge_triples_data_N_1}, the N_to_1 allocation is {allocation_knowledge_triples_data_N_N}.")
        
        allocations_allowed = [count_knowledge_triples_data_1_1, count_knowledge_triples_data_1_N, count_knowledge_triples_data_N_1, count_knowledge_triples_data_N_N]
        
        allocations = [allocation_knowledge_triples_data_1_1, allocation_knowledge_triples_data_1_N, allocation_knowledge_triples_data_N_1, allocation_knowledge_triples_data_N_N]
        
        total_allocation = sum(allocations)
        
        print(f"The currewnt total allocation is {total_allocation}.")

        while(total_allocation < total_sample_size):
            print(f"The current allocations is {allocations}. Less than total sample size {total_sample_size}.")
            condition_results = [allocation < avg_allocation for allocation in allocations]
            
            num_group_to_be_added = num_group - sum(condition_results)
            
            print(f"The num of relationship type need added is {num_group_to_be_added}.")
            
            less_indices = [index for index, condition_result in enumerate(condition_results) if condition_result]
            
            print(f"The less_indices is {less_indices}")
            
            new_avg_allocation = int((total_sample_size - sum([allocations[index] for index in less_indices])) / num_group_to_be_added)
            
            avg_allocation = new_avg_allocation
            
            print(f"The new avg allocation should be {new_avg_allocation}")
            
            allocations = [allocation if index in less_indices else new_avg_allocation for index, allocation in enumerate(allocations)]
            
            allocations = [allocation if allocation < allocations_allowed[index] else allocations_allowed[index] for index, allocation in enumerate(allocations)]
            
            if total_sample_size - sum(allocations) <= 4:
                difference_temp = total_sample_size - sum(allocations)
                for _ in range(difference_temp):
                    while True:
                        # 随机挑选一个索引
                        random_index = random.randint(0, len(allocations) - 1)
                        # 如果该索引对应的allocation小于其所允许的最大值
                        if allocations[random_index] < allocations_allowed[random_index]:
                            # 就在这个位置分配加1
                            allocations[random_index] += 1
                            break  # 成功分配后退出内循环
                    
                
            print(f"After updated, the allocation should be {allocations}")
            
            total_allocation = sum(allocations)
        
        while(total_allocation > total_sample_size):
            index_of_max = allocations.index(max(allocations))
            allocations[index_of_max] -= (total_allocation - total_sample_size)
            total_allocation = sum(allocations)        
        
        allocation_knowledge_triples_data_list = allocations
        
        allocation_knowledge_triples_data_1_1, allocation_knowledge_triples_data_1_N, allocation_knowledge_triples_data_N_1, allocation_knowledge_triples_data_N_N = allocations
        
        print(f"The allocation for 1:1 is {allocation_knowledge_triples_data_1_1}. The allocation for 1:N is {allocation_knowledge_triples_data_1_N}. The allocation for N:1 is {allocation_knowledge_triples_data_N_1}. The allocation for N:N is {allocation_knowledge_triples_data_N_N}.")
        
        relation2count_knowledge_triples_data_1_to_1 = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_1_to_1']
        relation2count_knowledge_triples_data_1_to_N = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_1_to_N']
        relation2count_knowledge_triples_data_N_to_1 = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_N_to_1']
        relation2count_knowledge_triples_data_N_to_N = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_N_to_N']
        
        relation2count_knowledge_triples_data_list = [relation2count_knowledge_triples_data_1_to_1, relation2count_knowledge_triples_data_1_to_N, relation2count_knowledge_triples_data_N_to_1, relation2count_knowledge_triples_data_N_to_N]
        
        relation2knowledge_triples_data_1_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_1']
        relation2knowledge_triples_data_1_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_N']
        relation2knowledge_triples_data_N_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_1']
        relation2knowledge_triples_data_N_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_N']
        
        relation2knowledge_triples_data_list = [relation2knowledge_triples_data_1_to_1, relation2knowledge_triples_data_1_to_N, relation2knowledge_triples_data_N_to_1, relation2knowledge_triples_data_N_to_N]
        relationship_type_list = ['1_to_1', '1_to_N', 'N_to_1', 'N_to_N']
        
        print("Then, we try to determine the allocation for each relation within 1:1, 1:N, N:1, N:N")
        
        for index, relation2knowledge_triples_data in enumerate(relation2knowledge_triples_data_list):
            relation2count_knowledge_triples_data = relation2count_knowledge_triples_data_list[index]
            allocation_knowledge_triples_data = allocation_knowledge_triples_data_list[index]
            count_triples_all_relations = sum(list(relation2count_knowledge_triples_data.values()))
            
            relationship_type = relationship_type_list[index]
            
            print(f"Processing the {relationship_type} ...")
            
            relation2allocation = {}
            
            for relation, count_knowledge_triples_data in relation2count_knowledge_triples_data.items():
                relation_triples_proportion = count_knowledge_triples_data / count_triples_all_relations
                relation2allocation[relation] = int(allocation_knowledge_triples_data * relation_triples_proportion)
            
            print(f"The current allocation for {relationship_type} is {relation2allocation}.")
            
            print(f"The sum of previous relation's allocations is {sum(list(relation2allocation.values()))}")
            
            print(f"The allowed allocation for {relationship_type} should be {allocation_knowledge_triples_data}")
            
            while(sum(list(relation2allocation.values())) != allocation_knowledge_triples_data):
                print(f"The sum of previous relation's allocations is {sum(list(relation2allocation.values()))}, not equal to the allowed allocation for {relationship_type} {allocation_knowledge_triples_data}")
                # random.seed(42)
                
                random_relation = random.choice(list(relation2allocation.keys()))
                print(f"The chosen random relation is {random_relation}")
                if sum(list(relation2allocation.values())) < allocation_knowledge_triples_data:
                    print(f"The relation to count_knowledge_triples_data dict should be \n{relation2count_knowledge_triples_data}")
                
                if sum(list(relation2allocation.values())) > allocation_knowledge_triples_data and relation2allocation[random_relation] > 0:
                    relation2allocation[random_relation] = relation2allocation[random_relation] - 1
                elif sum(list(relation2allocation.values())) < allocation_knowledge_triples_data and relation2allocation[random_relation] < relation2count_knowledge_triples_data[random_relation]:
                    relation2allocation[random_relation] = relation2allocation[random_relation] + 1
            
            # print(f"The allocation for 1:1 is {allocation_knowledge_triples_data_1_1}. The allocation for 1:N is {allocation_knowledge_triples_data_1_N}. The allocation for N:1 is {allocation_knowledge_triples_data_N_1}. The allocation for N:N is {allocation_knowledge_triples_data_N_N}.")
            samples_all_relations = []
            print("Sampling the data ...")
            for relation, knowledge_triples_data in relation2knowledge_triples_data.items():
                random.seed(42)
                # samples_N_to_1_single_relation = random.sample(knowledge_triples_data_N_to_1, count_samples_N_to_1)
                samples_single_relation = [
                    {**sample, 'relationship_type': relationship_type} for sample in random.sample(knowledge_triples_data, relation2allocation[relation])
                ]
                samples_all_relations.extend(samples_single_relation)
            
            samples.extend(samples_all_relations)
        
        return samples
    
    
    def stratified_sample_data(self, knowledge_triples_data, total_sample_size = 1250):
        random.seed(42)
        samples = []
        
        knowledge_triples_data_parsed_dict = self.parse_data(knowledge_triples_data)
        
        relations = knowledge_triples_data_parsed_dict['relations']
        count_relations = len(relations)
        
        relation2sample_allocation = {}
        total_count_knowledge_triples_data = knowledge_triples_data_parsed_dict['overall_count_knowledge_triples_data']
        
        if total_count_knowledge_triples_data <= total_sample_size:
            
            # for single_data in knowledge_triples_data:
            #     # todo 
            #     single_data.updata({'relationship_type': })
            
            # sampled_knowledge_triples_data = knowledge_triples_data
            # return sampled_knowledge_triples_data
            total_sample_size = total_count_knowledge_triples_data
            
            return knowledge_triples_data
        
        all_count_knowledge_triples_data = [knowledge_triples_data_parsed_dict[key] for key in ['overall_count_knowledge_triples_data_1_to_1', 'overall_count_knowledge_triples_data_1_to_N', 'overall_count_knowledge_triples_data_N_to_1', 'overall_count_knowledge_triples_data_N_to_N']]
        
        count_knowledge_triples_data_1_1, count_knowledge_triples_data_1_N, count_knowledge_triples_data_N_1, count_knowledge_triples_data_N_N = all_count_knowledge_triples_data
        
        # Decide allocation for total 1:1, 1:N, N:1, N:N
        print("********** Decide allocation for 1:1, 1:N, N:1, N:N **********")
        
        1250
        
        allocation_knowledge_triples_data_1_1 = int(total_sample_size * 0.32) # 400
        allocation_knowledge_triples_data_1_N = int(total_sample_size * 0.32) # 400
        allocation_knowledge_triples_data_N_1 = int(total_sample_size * 0.04) # 50
        allocation_knowledge_triples_data_N_N = int(total_sample_size * 0.32) # 400
        
        def assign_value_for_k_groups(total_value: int, max_value_groups: List[int]) -> List[int]:
            n_groups = len(max_value_groups)
            assert sum(max_value_groups) >= total_value, f"{sum(max_value_groups)} should not be less than {total_value}"
            
            assigned_values = [0] * n_groups  # Initialise allocation results
            remaining_value = total_value
            
            while remaining_value > 0:
                # Calculate the number of groups that have not reached the current maximum value
                active_groups = [i for i in range(n_groups) if assigned_values[i] < max_value_groups[i]]
                if not active_groups:
                    break  # If there are no more groups to allocate, exit the loop.
                
                avg_value = remaining_value // len(active_groups)  # Divide to ensure that each allocation is a whole number
                remainder = remaining_value % len(active_groups)  # Calculate the remaining values

                for i in active_groups:
                    # assigned_value is the lesser of avg_value and max_value_groups[i] - assigned_values[i]
                    additional_value = min(avg_value, max_value_groups[i] - assigned_values[i])
                    assigned_values[i] += additional_value
                    remaining_value -= additional_value
                
                # Residual portions are processed and allocated to the remaining allocable groups
                for i in active_groups[:remainder]:
                    if remaining_value > 0 and assigned_values[i] < max_value_groups[i]:
                        assigned_values[i] += 1
                        remaining_value -= 1

            return assigned_values
        
        allocation_knowledge_triples_data_1_1, allocation_knowledge_triples_data_1_N, allocation_knowledge_triples_data_N_1, allocation_knowledge_triples_data_N_N = assign_value_for_k_groups(total_value=total_sample_size, max_value_groups=[count_knowledge_triples_data_1_1, count_knowledge_triples_data_1_N, count_knowledge_triples_data_N_1, count_knowledge_triples_data_N_N])
        
        allocations = [allocation_knowledge_triples_data_1_1, allocation_knowledge_triples_data_1_N, allocation_knowledge_triples_data_N_1, allocation_knowledge_triples_data_N_N]
        
        print(f"The allocation for 1:1 is {allocation_knowledge_triples_data_1_1}. The allocation for 1:N is {allocation_knowledge_triples_data_1_N}. The allocation for N:1 is {allocation_knowledge_triples_data_N_1}. The allocation for N:N is {allocation_knowledge_triples_data_N_N}.")
        
        relation2count_knowledge_triples_data_1_to_1 = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_1_to_1']
        relation2count_knowledge_triples_data_1_to_N = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_1_to_N']
        relation2count_knowledge_triples_data_N_to_1 = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_N_to_1']
        relation2count_knowledge_triples_data_N_to_N = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_N_to_N']
        
        relation2count_knowledge_triples_data_list = [relation2count_knowledge_triples_data_1_to_1, relation2count_knowledge_triples_data_1_to_N, relation2count_knowledge_triples_data_N_to_1, relation2count_knowledge_triples_data_N_to_N]
        
        relation2knowledge_triples_data_1_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_1']
        relation2knowledge_triples_data_1_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_N']
        relation2knowledge_triples_data_N_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_1']
        relation2knowledge_triples_data_N_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_N']
        
        relation2knowledge_triples_data_list = [relation2knowledge_triples_data_1_to_1, relation2knowledge_triples_data_1_to_N, relation2knowledge_triples_data_N_to_1, relation2knowledge_triples_data_N_to_N]
        relationship_type_list = ['1_to_1', '1_to_N', 'N_to_1', 'N_to_N']
        
        allocation_knowledge_triples_data_list = allocations
        
        print("Then, we try to determine the allocation for each relation within 1:1, 1:N, N:1, N:N")
        
        for index, relation2knowledge_triples_data in enumerate(relation2knowledge_triples_data_list):
            relation2count_knowledge_triples_data = relation2count_knowledge_triples_data_list[index]
            allocation_knowledge_triples_data = allocation_knowledge_triples_data_list[index]
            count_triples_all_relations = sum(list(relation2count_knowledge_triples_data.values()))
            
            relationship_type = relationship_type_list[index]
            
            print(f"Processing the {relationship_type} ...")
            
            relation2allocation = {}
            
            for relation, count_knowledge_triples_data in relation2count_knowledge_triples_data.items():
                relation_triples_proportion = count_knowledge_triples_data / count_triples_all_relations
                relation2allocation[relation] = int(allocation_knowledge_triples_data * relation_triples_proportion)
            
            print(f"The current allocation for {relationship_type} is {relation2allocation}.")
            
            print(f"The sum of previous relation's allocations is {sum(list(relation2allocation.values()))}")
            
            print(f"The allowed allocation for {relationship_type} should be {allocation_knowledge_triples_data}")
            
            while(sum(list(relation2allocation.values())) != allocation_knowledge_triples_data):
                print(f"The sum of previous relation's allocations is {sum(list(relation2allocation.values()))}, not equal to the allowed allocation for {relationship_type} {allocation_knowledge_triples_data}")
                # random.seed(42)
                
                random_relation = random.choice(list(relation2allocation.keys()))
                print(f"The chosen random relation is {random_relation}")
                if sum(list(relation2allocation.values())) < allocation_knowledge_triples_data:
                    print(f"The relation to count_knowledge_triples_data dict should be \n{relation2count_knowledge_triples_data}")
                
                if sum(list(relation2allocation.values())) > allocation_knowledge_triples_data and relation2allocation[random_relation] > 0:
                    relation2allocation[random_relation] = relation2allocation[random_relation] - 1
                elif sum(list(relation2allocation.values())) < allocation_knowledge_triples_data and relation2allocation[random_relation] < relation2count_knowledge_triples_data[random_relation]:
                    relation2allocation[random_relation] = relation2allocation[random_relation] + 1
            
            # print(f"The allocation for 1:1 is {allocation_knowledge_triples_data_1_1}. The allocation for 1:N is {allocation_knowledge_triples_data_1_N}. The allocation for N:1 is {allocation_knowledge_triples_data_N_1}. The allocation for N:N is {allocation_knowledge_triples_data_N_N}.")
            samples_all_relations = []
            print("Sampling the data ...")
            for relation, knowledge_triples_data in relation2knowledge_triples_data.items():
                random.seed(42)
                # samples_N_to_1_single_relation = random.sample(knowledge_triples_data_N_to_1, count_samples_N_to_1)
                samples_single_relation = [
                    {**sample, 'relationship_type': relationship_type} for sample in random.sample(knowledge_triples_data, relation2allocation[relation])
                ]
                samples_all_relations.extend(samples_single_relation)
            
            samples.extend(samples_all_relations)
        
        return samples
    
    
    def stratified_sample_data_without_1_N(self, knowledge_triples_data, total_sample_size = 1000):
        random.seed(42)
        samples = []
        
        knowledge_triples_data_parsed_dict = self.parse_data(knowledge_triples_data)
        
        relations = knowledge_triples_data_parsed_dict['relations']
        count_relations = len(relations)
        
        relation2sample_allocation = {}
        total_count_knowledge_triples_data = knowledge_triples_data_parsed_dict['overall_count_knowledge_triples_data']
        
        if total_count_knowledge_triples_data <= total_sample_size:
            
            # for single_data in knowledge_triples_data:
            #     # todo 
            #     single_data.updata({'relationship_type': })
            
            # sampled_knowledge_triples_data = knowledge_triples_data
            # return sampled_knowledge_triples_data
            total_sample_size = total_count_knowledge_triples_data
        
        all_count_knowledge_triples_data = [knowledge_triples_data_parsed_dict[key] for key in ['overall_count_knowledge_triples_data_1_to_1', 'overall_count_knowledge_triples_data_1_to_N', 'overall_count_knowledge_triples_data_N_to_1', 'overall_count_knowledge_triples_data_N_to_N']]
        
        count_knowledge_triples_data_1_1, count_knowledge_triples_data_1_N, count_knowledge_triples_data_N_1, count_knowledge_triples_data_N_N = all_count_knowledge_triples_data
        
        # Decide allocation for total 1:1, 1:N, N:1, N:N
        print("********** Decide allocation for 1:1, 1:N, N:1, N:N **********")
        
        allocation_knowledge_triples_data_1_1 = int(total_sample_size * 0.334) # 334
        allocation_knowledge_triples_data_1_N = int(total_sample_size * 0) # 0
        allocation_knowledge_triples_data_N_1 = int(total_sample_size * 0.333) # 333
        allocation_knowledge_triples_data_N_N = int(total_sample_size * 0.333) # 333
        
        allocations = [allocation_knowledge_triples_data_1_1, allocation_knowledge_triples_data_1_N, allocation_knowledge_triples_data_N_1, allocation_knowledge_triples_data_N_N]
        
        print(f"The allocation for 1:1 is {allocation_knowledge_triples_data_1_1}. The allocation for 1:N is {allocation_knowledge_triples_data_1_N}. The allocation for N:1 is {allocation_knowledge_triples_data_N_1}. The allocation for N:N is {allocation_knowledge_triples_data_N_N}.")
        
        relation2count_knowledge_triples_data_1_to_1 = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_1_to_1']
        relation2count_knowledge_triples_data_1_to_N = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_1_to_N']
        relation2count_knowledge_triples_data_N_to_1 = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_N_to_1']
        relation2count_knowledge_triples_data_N_to_N = knowledge_triples_data_parsed_dict['relation2count_knowledge_triples_data_N_to_N']
        
        relation2count_knowledge_triples_data_list = [relation2count_knowledge_triples_data_1_to_1, relation2count_knowledge_triples_data_1_to_N, relation2count_knowledge_triples_data_N_to_1, relation2count_knowledge_triples_data_N_to_N]
        
        relation2knowledge_triples_data_1_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_1']
        relation2knowledge_triples_data_1_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_1_to_N']
        relation2knowledge_triples_data_N_to_1 = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_1']
        relation2knowledge_triples_data_N_to_N = knowledge_triples_data_parsed_dict['relation2knowledge_triples_data_N_to_N']
        
        relation2knowledge_triples_data_list = [relation2knowledge_triples_data_1_to_1, relation2knowledge_triples_data_1_to_N, relation2knowledge_triples_data_N_to_1, relation2knowledge_triples_data_N_to_N]
        relationship_type_list = ['1_to_1', '1_to_N', 'N_to_1', 'N_to_N']
        
        allocation_knowledge_triples_data_list = allocations
        
        print("Then, we try to determine the allocation for each relation within 1:1, 1:N, N:1, N:N")
        
        for index, relation2knowledge_triples_data in enumerate(relation2knowledge_triples_data_list):
            relation2count_knowledge_triples_data = relation2count_knowledge_triples_data_list[index]
            allocation_knowledge_triples_data = allocation_knowledge_triples_data_list[index]
            count_triples_all_relations = sum(list(relation2count_knowledge_triples_data.values()))
            
            relationship_type = relationship_type_list[index]
            
            print(f"Processing the {relationship_type} ...")
            
            relation2allocation = {}
            
            for relation, count_knowledge_triples_data in relation2count_knowledge_triples_data.items():
                relation_triples_proportion = count_knowledge_triples_data / count_triples_all_relations
                relation2allocation[relation] = int(allocation_knowledge_triples_data * relation_triples_proportion)
            
            print(f"The current allocation for {relationship_type} is {relation2allocation}.")
            
            print(f"The sum of previous relation's allocations is {sum(list(relation2allocation.values()))}")
            
            print(f"The allowed allocation for {relationship_type} should be {allocation_knowledge_triples_data}")
            
            while(sum(list(relation2allocation.values())) != allocation_knowledge_triples_data):
                print(f"The sum of previous relation's allocations is {sum(list(relation2allocation.values()))}, not equal to the allowed allocation for {relationship_type} {allocation_knowledge_triples_data}")
                # random.seed(42)
                
                random_relation = random.choice(list(relation2allocation.keys()))
                print(f"The chosen random relation is {random_relation}")
                if sum(list(relation2allocation.values())) < allocation_knowledge_triples_data:
                    print(f"The relation to count_knowledge_triples_data dict should be \n{relation2count_knowledge_triples_data}")
                
                if sum(list(relation2allocation.values())) > allocation_knowledge_triples_data and relation2allocation[random_relation] > 0:
                    relation2allocation[random_relation] = relation2allocation[random_relation] - 1
                elif sum(list(relation2allocation.values())) < allocation_knowledge_triples_data and relation2allocation[random_relation] < relation2count_knowledge_triples_data[random_relation]:
                    relation2allocation[random_relation] = relation2allocation[random_relation] + 1
            
            # print(f"The allocation for 1:1 is {allocation_knowledge_triples_data_1_1}. The allocation for 1:N is {allocation_knowledge_triples_data_1_N}. The allocation for N:1 is {allocation_knowledge_triples_data_N_1}. The allocation for N:N is {allocation_knowledge_triples_data_N_N}.")
            samples_all_relations = []
            print("Sampling the data ...")
            for relation, knowledge_triples_data in relation2knowledge_triples_data.items():
                random.seed(42)
                # samples_N_to_1_single_relation = random.sample(knowledge_triples_data_N_to_1, count_samples_N_to_1)
                samples_single_relation = [
                    {**sample, 'relationship_type': relationship_type} for sample in random.sample(knowledge_triples_data, relation2allocation[relation])
                ]
                samples_all_relations.extend(samples_single_relation)
            
            samples.extend(samples_all_relations)
        
        return samples
        
        
    def process_and_save_sample_knowledge_triples_data(self, total_sample_size = 1000, sample_type: str = 'dc'):
        random.seed(42)
        
        frequencies = list(self.snomedCT_knowledge_triples_dataset.frequency_dict.keys())
        
        sample_fn_dict = {
            'dc': self.dc_sample_data,
            'eq': self.eq_sample_data,
            'stratified': self.stratified_sample_data,
            'stratified_without_1_N': self.stratified_sample_data_without_1_N
        }
        
        assert sample_type in sample_fn_dict.keys(), f'sample_type is not supported, it should be within {sample_fn_dict.keys()}'
        
        sample_data_fn = sample_fn_dict[sample_type]
        
        sampled_knowledge_triples_data = []
        for frequency in frequencies:
            knowledge_triples_data = self.snomedCT_knowledge_triples_dataset.get_umls_knowledge_triples_data(frequency_type=frequency)
            print(f"Sampling {frequency} data ...")
            sampled_knowledge_triples_data_single = sample_data_fn(knowledge_triples_data=knowledge_triples_data, total_sample_size=total_sample_size)
            print(f"The num of orginal {frequency} triples is {len(knowledge_triples_data)} and the num of sampled {frequency} triples is {len(sampled_knowledge_triples_data_single)}\n")
            sampled_knowledge_triples_data.extend(sampled_knowledge_triples_data_single)
        
        sampled_knowledge_triples_data_sorted = sorted(sampled_knowledge_triples_data, key=lambda x: x['index'])
        
        # for i, sampled_knowledge_triple_data in enumerate(sampled_knowledge_triples_data_sorted):
        #     sampled_knowledge_triple_data['index'] = i
        
        save_json(sampled_knowledge_triples_data_sorted, f'clinic_knowledge_triples_{sample_type}_sampled.json', use_indent=True)
        
    


    def test(self, total_sample_size = 100):
        knowledge_triples_data_long_tail = self.snomedCT_knowledge_triples_dataset.get_umls_knowledge_triples_data(frequency_type='long_tail')
        print("Sampling long_tail data ...")
        sampled_knowledge_triples_data_long_tail = self.dc_sample_data(knowledge_triples_data=knowledge_triples_data_long_tail, total_sample_size=total_sample_size)
        print(f"The num of orginal long_tail triples is {len(knowledge_triples_data_long_tail)} and the num of sampled long_tail triples is {len(sampled_knowledge_triples_data_long_tail)}\n")
        sampled_knowledge_triples_data = []
        sampled_knowledge_triples_data.extend(sampled_knowledge_triples_data_long_tail)
        sampled_knowledge_triples_data_sorted = sorted(sampled_knowledge_triples_data, key=lambda x: x['index'])



# def update_NN_relationship_type_to_original_data(self, knowledge_triples_data)
def generte_triples_data_updated():
    frequency_dict = {
                'long_tail': (1,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None),
            }
    dataset = SnomedCTKnowledgeTriplesDataset(frequency_type='_', save_data_file_name='clinic_knowledge_triples.json', frequency_dict=frequency_dict, is_sample=False)
    filter = UMLSKnowledgeTriplesDataFilter(snomedCT_knowledge_triples_dataset=dataset)
    filter.update_NN_relationship_type_to_original_data(dataset.get_umls_knowledge_triples_data('all'))
    

def dc_sample_data():
    """
    10^0, 10^1, 10^2, 10^3, 
    """
    frequency_dict = {
                'long_tail': (1,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None),
            }
    dataset = SnomedCTKnowledgeTriplesDataset(frequency_type='_', save_data_file_name='clinic_knowledge_triples.json', frequency_dict=frequency_dict)
    filter = UMLSKnowledgeTriplesDataFilter(snomedCT_knowledge_triples_dataset=dataset)
    filter.process_and_save_sample_knowledge_triples_data(total_sample_size=1000, sample_type='dc')
    # filter.test()

def eq_sample_data():
    """
    10^0, 10^1, 10^2, 10^3, 
    """
    # frequency_dict = {
    #             'long_tail': (0,1),
    #             'medium': (2,10),
    #             'high': (11, 100),
    #             'very_high': (101, 1000),
    #             'popular': (1001, None)
    #         }
    frequency_dict = {
                'long_tail': (1,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None),
            }
    dataset = SnomedCTKnowledgeTriplesDataset(frequency_type='_', save_data_file_name='clinic_knowledge_triples.json', frequency_dict=frequency_dict, is_sample=False)
    filter = UMLSKnowledgeTriplesDataFilter(snomedCT_knowledge_triples_dataset=dataset)
    filter.process_and_save_sample_knowledge_triples_data(total_sample_size=1000, sample_type='eq')
    # filter.test()


def stratified_sample_data():
    """
    0, 10^0, 10^1, 10^2, 10^3, 
    """
    frequency_dict = {
                'long_tail': (1,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None),
            }
    # frequency_dict = {
    #             'long_tail': (0,1),
    #             'medium': (1,10),
    #             'high': (10, 100),
    #             'very_high': (100, None),
    #         }
    dataset = SnomedCTKnowledgeTriplesDataset(frequency_type='_', save_data_file_name='clinic_knowledge_triples.json', frequency_dict=frequency_dict, is_sample=False)
    filter = UMLSKnowledgeTriplesDataFilter(snomedCT_knowledge_triples_dataset=dataset)
    filter.process_and_save_sample_knowledge_triples_data(total_sample_size=1000, sample_type='stratified')
    # filter.test()


def stratified_sample_data_without_1_N():
    """
    0, 10^0, 10^1, 10^2, 10^3, 
    """
    # frequency_dict = {
    #             'long_tail': (0,1),
    #             'medium': (2,10),
    #             'high': (11, 100),
    #             'very_high': (101, 1000),
    #             'popular': (1001, None)
    #         }
    frequency_dict = {
                'long_tail': (1,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None),
            }
    dataset = SnomedCTKnowledgeTriplesDataset(frequency_type='_', save_data_file_name='clinic_knowledge_triples.json', frequency_dict=frequency_dict, is_sample=False)
    filter = UMLSKnowledgeTriplesDataFilter(snomedCT_knowledge_triples_dataset=dataset)
    filter.process_and_save_sample_knowledge_triples_data(total_sample_size=1000, sample_type='stratified_without_1_N')
    # filter.test()
    
def sample_data():
    ret_loader = RetLoader()
    # index2score = ret_loader.load_index2score_pickle_file(model_name='BioMedLM', frequency='long_tail', prompt_mode=0, is_demonstration=True, editing_method='ROME', sampling_type='stratified_sampling')
        
    def sample(ranked_indexes: List, tails_num, tail_sample_num, total_sample_num:int=1000):
        tails = ranked_indexes[-tails_num:]
        sample_from_tails = random.sample(tails, tail_sample_num)
        
        remaining_data = ranked_indexes[:-tails_num]
        sample_from_remaining = random.sample(remaining_data, total_sample_num - tail_sample_num)
        
        # 合并两个抽样结果，构成最终的 1000 个样本
        final_sampled_indexes = sample_from_tails + sample_from_remaining
        
        return final_sampled_indexes
    
    def extract_by_indexes(index2score: dict, new_indexes: list):
        # 构造新的字典，包含在 popular_indexes 中的键值对
        extracted_dict = {index: index2score[index] for index in new_indexes if index in index2score}
        
        return extracted_dict
    
        # long_tail
    long_tail_index2score = ret_loader.load_index2score_pickle_file(model_name='BioMedLM', frequency='long_tail', prompt_mode=0, is_demonstration=True)
    long_tail_ranked_indexes = ret_loader.rank_index2score(long_tail_index2score, is_edit_score=False)
    long_tail_indexes = sample(ranked_indexes=long_tail_ranked_indexes, tails_num=8500, tail_sample_num=295)
    sampled_long_tail_index2score = extract_by_indexes(index2score=long_tail_index2score, new_indexes=long_tail_indexes)
    print(f"long-tail: {ret_loader.avg_index2score(sampled_long_tail_index2score, is_edit_score=False)}")
    
    medium_index2score = ret_loader.load_index2score_pickle_file(model_name='BioMedLM', frequency='medium', prompt_mode=0, is_demonstration=True)
    medium_ranked_indexes = ret_loader.rank_index2score(medium_index2score, is_edit_score=False)
    medium_indexes = sample(ranked_indexes=medium_ranked_indexes, tails_num=2500, tail_sample_num=255)
    sampled_medium_index2score = extract_by_indexes(index2score=medium_index2score, new_indexes=medium_indexes)
    print(f"medium: {ret_loader.avg_index2score(sampled_medium_index2score, is_edit_score=False)}")
    
    high_index2score = ret_loader.load_index2score_pickle_file(model_name='BioMedLM', frequency='high', prompt_mode=0, is_demonstration=True)
    high_ranked_indexes = ret_loader.rank_index2score(high_index2score, is_edit_score=False)
    high_indexes = sample(ranked_indexes=high_ranked_indexes, tails_num=700, tail_sample_num=212)
    sampled_high_index2score = extract_by_indexes(index2score=high_index2score, new_indexes=high_indexes)
    print(f"high: {ret_loader.avg_index2score(sampled_high_index2score, is_edit_score=False)}")
    
    popular_index2score = ret_loader.load_index2score_pickle_file(model_name='BioMedLM', frequency='popular', prompt_mode=0, is_demonstration=True)
    popular_ranked_indexes = ret_loader.rank_index2score(popular_index2score, is_edit_score=False)
    popular_indexes = sample(ranked_indexes=popular_ranked_indexes, tails_num=145, tail_sample_num=88, total_sample_num=500)
    sampled_popular_index2score = extract_by_indexes(index2score=popular_index2score, new_indexes=popular_indexes)
    print(f"popular: {ret_loader.avg_index2score(sampled_popular_index2score, is_edit_score=False)}")
    
    
    sampled_indexes = long_tail_indexes + medium_indexes + high_indexes + popular_indexes
    
    frequency_dict = {
                'long_tail': (1,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None)
            }
    clikt_dataset = SnomedCTKnowledgeTriplesDataset(frequency_type="all", frequency_dict=frequency_dict)
    
    sampled_knowledge_triple_data = clikt_dataset.load_knowledge_triple_data_from_data_indexes(data_indexes=sampled_indexes)
    
    save_file_name = "clinic_knowledge_triples_sampled.json"
    # unbalanced_one_to_many_knowledge_list_example_file_name = "unbalanced_one_to_many_knowledge_list_example.json"
    save_path = save_file_name
    
    sampled_knowledge_triple_data = sorted(sampled_knowledge_triple_data, key=lambda x: x["index"])
    
    
    save_json(data=sampled_knowledge_triple_data, path=save_path, use_indent=True)

    
    
def analyse_all_data():
    frequency_dict = {
                'long_tail': (0,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None),
            }
    dataset = SnomedCTKnowledgeTriplesDataset(frequency_type='_', save_data_file_name='clinic_knowledge_triples.json', frequency_dict=frequency_dict, is_sample=True)
    filter = UMLSKnowledgeTriplesDataFilter(snomedCT_knowledge_triples_dataset=dataset)
    filter.analyse_all_data(is_sampled_data=True)

def analyse_all_updted_data():
    frequency_dict = {
                'long_tail': (1,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None),
            }
    dataset = SnomedCTKnowledgeTriplesDataset(frequency_type='_', save_data_file_name='clinic_knowledge_triples_updated.json', frequency_dict=frequency_dict, is_sample=True)
    filter = UMLSKnowledgeTriplesDataFilter(snomedCT_knowledge_triples_dataset=dataset)
    filter.analyse_all_data(is_sampled_data=True)
    
def analyse_sampled_data():
    frequency_dict = {
                'long_tail': (1,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None)
            }
    sampled_dataset = SnomedCTKnowledgeTriplesDataset(frequency_type='_', save_data_file_name='clinic_knowledge_triples_sampled.json', frequency_dict=frequency_dict, is_sample=True)
    sampled_filter = UMLSKnowledgeTriplesDataFilter(snomedCT_knowledge_triples_dataset=sampled_dataset)
    sampled_filter.analyse_all_data(is_sampled_data=True)


def analyse_dc_sampled_data():
    frequency_dict = {
                'long_tail': (1,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None)
            }
    sampled_dataset = SnomedCTKnowledgeTriplesDataset(frequency_type='_', save_data_file_name='clinic_knowledge_triples_dc_sampled.json', frequency_dict=frequency_dict, is_sample=True)
    sampled_filter = UMLSKnowledgeTriplesDataFilter(snomedCT_knowledge_triples_dataset=sampled_dataset)
    sampled_filter.analyse_all_data(is_sampled_data=True)


def analyse_eq_sampled_data():
    frequency_dict = {
                'long_tail': (1,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None)
            }
    sampled_dataset = SnomedCTKnowledgeTriplesDataset(frequency_type='_', save_data_file_name='clinic_knowledge_triples_eq_sampled.json', frequency_dict=frequency_dict, is_sample=True)
    sampled_filter = UMLSKnowledgeTriplesDataFilter(snomedCT_knowledge_triples_dataset=sampled_dataset)
    sampled_filter.analyse_all_data(is_sampled_data=True)
    
def analyse_stratified_sampled_data():
    frequency_dict = {
                'long_tail': (1,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None)
            }
    sampled_dataset = SnomedCTKnowledgeTriplesDataset(frequency_type='_', save_data_file_name='clinic_knowledge_triples_stratified_sampled.json', frequency_dict=frequency_dict, is_sample=True)
    sampled_filter = UMLSKnowledgeTriplesDataFilter(snomedCT_knowledge_triples_dataset=sampled_dataset)
    sampled_filter.analyse_all_data(is_sampled_data=True)
    
def analyse_stratified_sampled_data_without_1_N():
    frequency_dict = {
                'long_tail': (1,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None)
            }
    sampled_dataset = SnomedCTKnowledgeTriplesDataset(frequency_type='_', save_data_file_name='clinic_knowledge_triples_stratified_without_1_N_sampled.json', frequency_dict=frequency_dict, is_sample=True)
    sampled_filter = UMLSKnowledgeTriplesDataFilter(snomedCT_knowledge_triples_dataset=sampled_dataset)
    sampled_filter.analyse_all_data(is_sampled_data=True)



def generate_clinic_knowledge_triples_data():
    generator = UMLSKnowledgeTriplesDataGenerator()
    knowledge_triples_data = generator.generate()
    
    data_filter = UMLSKnowledgeTriplesDataFilter()
    # data_filter.analyse_all_data()
    
    filtered_knowledge_triples_data = data_filter.filter_data(knowledge_triples_data=knowledge_triples_data,
                                     is_filter_1_to_1=False, 
                                     is_filter_1_to_N=False, 
                                     is_filter_N_to_1=False, 
                                     is_filter_N_to_N=False, 
                                     relations_for_removal=[])
    
    clinic_knowledge_triples_data = data_filter.update_NN_relationship_type_to_original_data(filtered_knowledge_triples_data, is_save=False)
    
    
    clinic_knowledge_triples_data = data_filter.update_locality_data_to_original_data(knowledge_triples_data=clinic_knowledge_triples_data, is_save=False)
    
    # clinic_knowledge_triples_data = data_filter.filter_zero_coexist_num_data(knowledge_triples_data=clinic_knowledge_triples_data)
    
    clinic_knowledge_triples_data = data_filter.rebuild_data_index(knowledge_triples_data=clinic_knowledge_triples_data)
    
    
    
    
    save_json(data=clinic_knowledge_triples_data, path=os.path.join(save_dir_path, 'clinic_knowledge_triples.json'), use_indent=True)
    print(f"Successfully save data to \"{os.path.join(save_dir_path, 'clinic_knowledge_triples.json')}n\"")
    
    save_json(data=clinic_knowledge_triples_data[:10], path=os.path.join(save_dir_path, 'clinic_knowledge_triples_example.json'), use_indent=True)
    
    print(f"Successfully save data to \"{os.path.join(save_dir_path, 'clinic_knowledge_triples_example.json')}n\"")
    
    
if __name__ == "__main__":
    
    """
    10^0, 10^1, 10^2, 10^3, 
    """
        
    generate_clinic_knowledge_triples_data()
    
    # dc_sample_data()
    
    # eq_sample_data()
    
    # stratified_sample_data()
    
    # stratified_sample_data_without_1_N()
    
    # analyse_sampled_data()
    
    # analyse_eq_sampled_data()
    
    # analyse_dc_sampled_data()
    
    # analyse_stratified_sampled_data()
    
    # analyse_all_data()
    
    # analyse_stratified_sampled_data_without_1_N()