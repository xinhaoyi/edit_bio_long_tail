import sys
from typing import Dict, List, Tuple, Union

# sys.path.append("/nfs")
# sys.path.append("/nfs/general")
# sys.path.append("/nfs/long_tail")

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import random
from data.pubtator.pubtator_processor import MultiPubtatorProcessor
from dsets.pubtator_dataset import PubTatorDataset
from dsets.knowledge_graph_dataset import KnowledgeGraphDataset, SnomedCTDataset, UMLSDataset
from utils.my_utils import load_json, save_json
from config import path_config


import pickle 


class PubtatorLinkedToKGDataset:
    def __init__(self, save_data_path: str, knowledge_graph_dataset: KnowledgeGraphDataset) -> None:
        self.save_data_path = save_data_path
        self.pubtator_dataset = PubTatorDataset()
        self.__knowledge_graph_dataset = knowledge_graph_dataset
        self.pubtator_file_name = self.pubtator_dataset.file_name
    
    def load_entityid2mentions(self):
        id2PMIDs_path = os.path.join(self.save_data_path, f"{self.pubtator_file_name}_id2PMIDs.pickle")
        id2mentions = pickle.load(open(id2PMIDs_path, mode="rb"))
        return id2mentions
    
    def load_entityid2midxs(self):
        # midx 是 mention index
        id2midxs_path = os.path.join(self.save_data_path, f"{self.pubtator_file_name}_id2midxs.pickle")
        # id2midxs这里面id是字符串，midx是数字
        id2midxs = pickle.load(open(id2midxs_path, mode="rb"))
        return id2midxs
    
    def load_entityid2PMIDs(self):
        id2PMIDs_path = os.path.join(self.save_data_path, f"{self.pubtator_file_name}_id2PMIDs.pickle")
        id2PMIDs = pickle.load(open(id2PMIDs_path, mode="rb"))
        return id2PMIDs
    
    def load_midx2mention(self):
        midx2mention_path = os.path.join(self.save_data_path, f"{self.pubtator_file_name}_midx2mention.pickle")
        midx2mention = pickle.load(open(midx2mention_path, mode="rb"))
        return midx2mention
    
    def load_triple2coexistPMIDs(self):
        triple2coexistPMIDs_path = os.path.join(self.save_data_path, f"{self.pubtator_file_name}_triple2coexistPMIDs.pickle")
        triple2coexistPMIDs = pickle.load(open(triple2coexistPMIDs_path, mode="rb"))
        return triple2coexistPMIDs
    
    def load_triple2coexist_num(self):
        triple2coexistPMIDs = self.load_triple2coexistPMIDs()
        def convert_triple2coexistPMIDs_to_triple2coexist_num(triple2coexistPMIDs: dict):
            triple2coexist_num = {}
            for triple, PMIDs in triple2coexistPMIDs.items():
                triple2coexist_num[triple] = len(PMIDs)
            return triple2coexist_num
        triple2coexist_num = convert_triple2coexistPMIDs_to_triple2coexist_num(triple2coexistPMIDs)
        return triple2coexist_num
    
    def load_remaining_triples(self):
        remaining_triples_file_path = os.path.join(self.save_data_path, f"{self.pubtator_file_name}_remaining_triples.pickle")
        if not os.path.exists(remaining_triples_file_path):
            triple2coexist_num = self.load_triple2coexist_num()
            remaining_triples = list(triple2coexist_num.keys())
            print(f"Saving remaining triples to {remaining_triples_file_path} ...")
            pickle.dump(remaining_triples, open(remaining_triples_file_path, mode='wb'))
            print(f"Successful Saving in {remaining_triples_file_path} and return the remaining triples")
        else:
            print(f"Loading remaining triples from {remaining_triples_file_path}")
            remaining_triples = pickle.load(open(remaining_triples_file_path, mode='rb'))
        return remaining_triples
    
    def load_remaining_triple2triple_with_real_name(self):
        remaining_triple2triple_with_real_name_file_path = os.path.join(self.save_data_path, f"{self.pubtator_file_name}_remaining_triple2triple_name.pickle")
        
        if not os.path.exists(remaining_triple2triple_with_real_name_file_path):
            entity2id = self.__knowledge_graph_dataset.load_entity2id()
            id2entityname = self.__knowledge_graph_dataset.load_id2entityname()
            relation2id = self.__knowledge_graph_dataset.load_relation2id()
            id2relationname = self.__knowledge_graph_dataset.load_id2relationname()
            triple2triple_with_real_name = {}
            remaining_triples = self.load_remaining_triples()
            for triple in remaining_triples:
                subject_entity = triple[0]
                relation = triple[1]
                object_entity = triple[2]
                
                subject_entity_name = id2entityname[entity2id[subject_entity]]
                relation_name = id2relationname[relation2id[relation]]
                object_entity_name = id2entityname[entity2id[object_entity]]
                
                triple_with_real_name = (subject_entity_name, relation_name, object_entity_name)
                
                triple2triple_with_real_name[triple] = triple_with_real_name
            print(f"Saving remaining triples to {remaining_triple2triple_with_real_name_file_path} ...")
            pickle.dump(triple2triple_with_real_name, open(remaining_triple2triple_with_real_name_file_path, mode='wb'))
            print(f"Successful Saving in {remaining_triple2triple_with_real_name_file_path} and return the remaining_triple to triple_with_real_name dictionary")
        else:
            print(f"Loading remaining triples from {remaining_triple2triple_with_real_name_file_path}")
            triple2triple_with_real_name = pickle.load(open(remaining_triple2triple_with_real_name_file_path, mode='rb'))
        
        return triple2triple_with_real_name
    
        
    def load_remaining_triples_with_real_names(self):
        remaining_triple2triple_with_real_name = self.load_remaining_triple2triple_with_real_name()
        triples_with_real_name = remaining_triple2triple_with_real_name.values()
        return triples_with_real_name
    
    
    def load_all_relation_data(self, is_load_static=True):
        """
        relation_data:
            'relation':relationname2relation[relationname],
            'relation name': relationname, 
            'triples_num': num, 
            'triples': relationname2triples[relationname]
            the triples here is acyally the "triples_with_real_name"
        """
        relation_data_list_file_path = os.path.join(self.save_data_path, f"{self.pubtator_file_name}_relation_data_list.pickle")
        
        if os.path.exists(relation_data_list_file_path) and is_load_static:
            print(f"Loading remaining triples from {relation_data_list_file_path}")
            sorted_relation_data_list = pickle.load(open(relation_data_list_file_path, mode='rb'))
        
        else:
            triple2coexist_num = self.load_triple2coexist_num()
            
            relation2id = self.__knowledge_graph_dataset.load_relation2id()
            id2relationname = self.__knowledge_graph_dataset.load_id2relationname()
            
            entity2id = self.__knowledge_graph_dataset.load_entity2id()
            id2entityname = self.__knowledge_graph_dataset.load_id2entityname()
            
            relationname2triples_num = {}
            relationname2relation = {}
            relationname2triples = {}
            relationname2triple_examples = {}
            
            for subject_entity, relation, object_entity in triple2coexist_num.keys():
                relationname = id2relationname[relation2id[relation]]
                
                subject_entity_name = id2entityname[entity2id[subject_entity]]
                object_entity_name = id2entityname[entity2id[object_entity]]
                
                # triple_with_real_name = {'subject_entity':subject_entity_name, 'relation': relationname, 'object_entity':object_entity_name}
                triple_with_real_name = (subject_entity_name, relationname, object_entity_name)
                triple_example = {'subject_entity':subject_entity_name, 'relation': relationname, 'object_entity':object_entity_name}
                
                relationname2triples_num[relationname] = relationname2triples_num.get(relationname, 0) + 1
                if relationname not in relationname2relation.keys():
                    relationname2relation[relationname] = relation
                    
                # relationname2triple_case[relationname] = relationname2triple_case.get(relationname, [])
                relationname2triples.setdefault(relationname, []).append(triple_with_real_name)
                relationname2triple_examples.setdefault(relationname, []).append({'triple':triple_example})
                
                
            relation_data_list = []
            relation_data_example_list = []
            for relationname, num in relationname2triples_num.items():
                relation_data_list.append({'relation':relationname2relation[relationname], 'relation name': relationname, 'triples_num': num, 'triples': relationname2triples[relationname]})
                relation_data_example_list.append({'relation':relationname2relation[relationname], 'relation name': relationname, 'triples_num': num, 'cases': relationname2triple_examples[relationname][:10]})
                
            sorted_relation_data_list = sorted(relation_data_list, key=lambda x: x['triples_num'], reverse=True)
            print(f"The total num of relations is {len(sorted_relation_data_list)}")
            
            # sorted_relation_data_example_list = sorted(relation_data_example_list, key=lambda x: x['triples_num'], reverse=True)
            print(f"Saving relation data list to {relation_data_list_file_path} ...")
            pickle.dump(sorted_relation_data_list, open(relation_data_list_file_path, mode='wb'))
            # save_json(path=os.path.join(self.save_data_path, 'relation_data_example_list.json'), data=sorted_relation_data_example_list, use_indent=True)
        
        return sorted_relation_data_list
        
        
    def load_remaining_relations(self):
        relation_data_list = self.load_all_relation_data()        
        relations = [relation_data['relation'] for relation_data in relation_data_list]
        return relations
    
    
    def load_remaining_relationnames(self):
        relation_data_list = self.load_all_relation_data()
        relationnames = [relation_data['relation name'] for relation_data in relation_data_list]
        return relationnames
    
    def load_relation2triples_distribution(self, is_load_static=True):
        """
        A -> 1
        A -> 2
        B -> 1
        """
        relation2triples_distribution_file_path = os.path.join(self.save_data_path, f"{self.pubtator_file_name}_relation2triples_distribution.pickle")
        if os.path.exists(relation2triples_distribution_file_path) and is_load_static:
            print(f"Loading relation2triples_distribution_data from {relation2triples_distribution_file_path}")
            relation2triples_distribution_data = pickle.load(open(relation2triples_distribution_file_path, mode='rb'))
            
        else:
            relation_data_list = self.load_all_relation_data()
            relation2triples_distribution_data: dict[str, Union(str, List[Tuple], List[Tuple], List[Tuple], List[Tuple])] = {} # type: ignore
            for relation_data in relation_data_list:
                relation = relation_data['relation']
                relation_name = relation_data['relation']
                triples_with_real_name = relation_data['triples']
                
                head_entity_name2count = dict()
                tail_entity_name2count = dict()
                
                # head_entityname2triples_single_to_N: dict[str: List[Tuple]] = dict()
                # tail_entityname2triples_N_to_single: dict[str: List[Tuple]] = dict()
                triples_1_to_1 = []
                triples_1_to_N = []
                triples_N_to_1 = []
                triples_N_to_N = []
                
                for triple_with_real_name in triples_with_real_name:
                    head_entity_name = triple_with_real_name[0]
                    tail_entity_name = triple_with_real_name[2]
                    
                    head_entity_name2count[head_entity_name] = head_entity_name2count.get(head_entity_name, 0) + 1
                    tail_entity_name2count[tail_entity_name] = tail_entity_name2count.get(tail_entity_name, 0) + 1
                
                for triple_with_real_name in triples_with_real_name:
                    head_entity_name = triple_with_real_name[0]
                    tail_entity_name = triple_with_real_name[2]
                    if head_entity_name2count[head_entity_name] > 1 and tail_entity_name2count[tail_entity_name] == 1:
                        """
                        A -> 1
                        A -> 2
                        """
                        # head_entityname2triples_single_to_N.setdefault(head_entity_name, []).append(triple_with_real_name)
                        triples_1_to_N.append(triple_with_real_name)
                    elif head_entity_name2count[head_entity_name] == 1 and tail_entity_name2count[tail_entity_name] > 1:
                        """
                        A -> 1
                        B -> 1
                        """
                        # tail_entityname2triples_N_to_single.setdefault(tail_entity_name, []).append(triple_with_real_name)
                        triples_N_to_1.append(triple_with_real_name)
                    elif head_entity_name2count[head_entity_name] == 1 and tail_entity_name2count[tail_entity_name] == 1:
                        triples_1_to_1.append(triple_with_real_name)
                    else:
                        """
                        A -> 1
                        A -> 2
                        B -> 1
                        
                        A -> 1 is a N_to_N
                        """
                        triples_N_to_N.append(triple_with_real_name)
                    # triples_1_to_N = [triples_single_to_N for triples_single_to_N in head_entityname2triples_single_to_N.values()]
                    # triples_N_to_1 = [triples_single_to_N for triples_single_to_N in tail_entityname2triples_N_to_single.values()]
                
                relation2triples_distribution_data[relation] = {'relation name': relation_name,
                                                            'triples_1_to_1': triples_1_to_1,
                                                            'triples_1_to_N': triples_1_to_N,
                                                            'triples_N_to_1': triples_N_to_1,
                                                            'triples_N_to_N': triples_N_to_N}
            print(f"Saving relation2triples_distribution_data to {relation2triples_distribution_file_path} ...")
            pickle.dump(relation2triples_distribution_data, open(relation2triples_distribution_file_path, mode='wb'))

        return relation2triples_distribution_data
    
    def generate_relation_example(self):
        relation_data_list = self.load_all_relation_data()
        relation2triples_distribution = self.load_relation2triples_distribution()
        data_list = []
        for relation_data in relation_data_list:
            relation = relation_data['relation']
            relationname = relation_data['relation name']
            triples_num = relation_data['triples_num']
            triples = relation_data['triples'][:10]
            
            n_triples_1_to_1 = len(relation2triples_distribution[relation]['triples_1_to_1'])
            n_triples_1_to_N = len(relation2triples_distribution[relation]['triples_1_to_N'])
            n_triples_N_to_1 = len(relation2triples_distribution[relation]['triples_N_to_1'])
            n_triples_N_to_N = len(relation2triples_distribution[relation]['triples_N_to_N'])
            
            data = {'relation': relation,
                    'relation_name': relationname,
                    'triples_num': triples_num,
                    'n_triples_1_to_1': f"{n_triples_1_to_1} ({(n_triples_1_to_1 / triples_num) * 100:.2f}%)",
                    'n_triples_1_to_N': f"{n_triples_1_to_N} ({(n_triples_1_to_N / triples_num) * 100:.2f}%)",
                    'n_triples_N_to_1': f"{n_triples_N_to_1} ({(n_triples_N_to_1 / triples_num) * 100:.2f}%)",
                    'n_triples_N_to_N': f"{n_triples_N_to_N} ({(n_triples_N_to_N / triples_num) * 100:.2f}%)",
                    'triples': triples
                    }
            data_list.append(data)
        
        save_json(path=os.path.join(self.save_data_path, 'relation_data_example_list.json'), data=data_list, use_indent=True)
        
    
    def draw_excell_based_on_relation_data_example(self):
        
        relation_data_example_list = load_json(os.path.join(self.save_data_path, 'relation_data_example_list.json'))
        
        df = pd.DataFrame(columns=["relation_name", "n_triples", "1_to_1_triples", "1_to_N_triples", "N_to_1_triples", "N_to_N_triples", "cases"])
        
        for relation_data_example in relation_data_example_list:
            cases_str = '\n'.join(['(' + ', '.join(triple) + ')' for triple in relation_data_example['triples']])
            item_dir = pd.DataFrame([{
                "relation_name": relation_data_example["relation_name"],
                "n_triples": relation_data_example["triples_num"],
                "1_to_1_triples": relation_data_example["n_triples_1_to_1"],
                "1_to_N_triples": relation_data_example["n_triples_1_to_N"],
                "N_to_1_triples": relation_data_example["n_triples_N_to_1"],
                "N_to_N_triples": relation_data_example["n_triples_N_to_N"],
                "cases": cases_str
            }])
            df = pd.concat([df, item_dir], ignore_index=True)
            
        excel_path = os.path.join(self.save_data_path, 'relation_data_example.xlsx')
        
        df.to_excel(excel_path, index=False)
                    

default_save_pubtator2snomedct_data_path = path_config.PUBTATOR_2_SNOMEDCT_DATA_DIR

class PubtatorLinkedToSnomedCTDataset(PubtatorLinkedToKGDataset):
    def __init__(self, save_data_path: str = default_save_pubtator2snomedct_data_path) -> None:
        super().__init__(save_data_path=save_data_path, knowledge_graph_dataset=SnomedCTDataset())
        
    
        
        

default_save_pubtator2umls_data_path = path_config.PUBTATOR_2_UMLS_DATA_DIR

class PubtatorLinkedToUMLSDataset(PubtatorLinkedToKGDataset):
    def __init__(self, save_data_path: str = default_save_pubtator2umls_data_path) -> None:
        super().__init__(save_data_path=save_data_path, knowledge_graph_dataset=UMLSDataset())
        


class FilterPubtatorLinkedToSnomedCTDataset():
    def __init__(self, 
                 pubtator_linked_snomedCT_dataset: PubtatorLinkedToSnomedCTDataset,
                 relation_triple_num_threshold=100,
                 relation_names_to_be_removed=[],
                 k_triples=5,
                 is_filter_process=False) -> None:
        self.pubtator_linked_snomedCT_dataset = pubtator_linked_snomedCT_dataset
        
        filtered_relation_data_list_file_path = os.path.join(self.pubtator_linked_snomedCT_dataset.save_data_path, 'filtered_relation_data_list.json')
        relation_name2k_triples_with_real_name_file_path = os.path.join(self.pubtator_linked_snomedCT_dataset.save_data_path, 'relation_name2k_triples_with_real_name.json')
        
        if is_filter_process or not any((os.path.exists(filtered_relation_data_list_file_path), os.path.exists(relation_name2k_triples_with_real_name_file_path))):
            self.__filter_process(relation_triple_num_threshold=relation_triple_num_threshold, relation_names_to_be_removed=relation_names_to_be_removed, k_triples=k_triples)
    
    def __filter_process(self, relation_triple_num_threshold=100, relation_names_to_be_removed = [], k_triples=5):
        filtered_relation_data_list = self.__filter_relations_with_threshold(relation_triple_num_threshold)
        filtered_relation_data_list = self.__filter_selected_relation_names(relation_data_list=filtered_relation_data_list, relation_names_to_be_removed=relation_names_to_be_removed)
        filtered_relation_data_list, relation_name2k_triples_with_real_name = self.__filter_k_triples_for_each_relation(relation_data_list=filtered_relation_data_list, k=k_triples)
        
        print(f"Saving filtered_relation_data_list to {os.path.join(self.pubtator_linked_snomedCT_dataset.save_data_path, 'filtered_relation_data_list.pickle')} ...")
        pickle.dump(filtered_relation_data_list, open(os.path.join(self.pubtator_linked_snomedCT_dataset.save_data_path, 'filtered_relation_data_list.pickle'), mode='wb'))

        
        
        print(f"Saving relation_name2k_triples_with_real_name to {os.path.join(self.pubtator_linked_snomedCT_dataset.save_data_path, 'relation_name2k_triples_with_real_name.pickle')} ...")
        save_json(path=os.path.join(self.pubtator_linked_snomedCT_dataset.save_data_path, 'relation_name2k_triples_with_real_name.json'), data=relation_name2k_triples_with_real_name, use_indent=True)
        pickle.dump(relation_name2k_triples_with_real_name, open(os.path.join(self.pubtator_linked_snomedCT_dataset.save_data_path, 'relation_name2k_triples_with_real_name.pickle'), mode='wb'))

    def load_filtered_relation_data_list(self):
        filtered_relation_data_list = pickle.load(open(os.path.join(self.pubtator_linked_snomedCT_dataset.save_data_path, 'filtered_relation_data_list.pickle'), mode='rb'))
        return filtered_relation_data_list
    
    def load_relation_name2k_triples_with_real_name(self):
        relation_name2k_triples_with_real_name = pickle.load(open(os.path.join(self.pubtator_linked_snomedCT_dataset.save_data_path, 'relation_name2k_triples_with_real_name.pickle'), mode='rb'))
        return relation_name2k_triples_with_real_name
    
    def load_filtered_triples_with_real_name(self):
        
        triples_with_real_name = []
        
        filtered_relation_data_list = self.load_filtered_relation_data_list()
        for relation_data in filtered_relation_data_list:
            triples_with_real_name_for_relation = relation_data['triples']
            triples_with_real_name.extend(triples_with_real_name_for_relation)
        
        return triples_with_real_name
    
    def load_filtered_triples(self):
        remaining_triple2triple_with_real_name = self.pubtator_linked_snomedCT_dataset.load_remaining_triple2triple_with_real_name()
        
        triple_with_real_name2triple = {}
        
        for triple, triple_with_real_name in remaining_triple2triple_with_real_name.items():
            triple_with_real_name2triple[triple_with_real_name] = triple
        
        filtered_triples = []
        
        filtered_triples_with_real_name = self.load_filtered_triples_with_real_name()
        for triple_with_real_name in filtered_triples_with_real_name:
            triple = triple_with_real_name2triple[triple_with_real_name]
            filtered_triples.append(triple)
        return filtered_triples
            
    
    def load_filtered_triple2coexist_num(self):
        original_triple2coexist_num = self.pubtator_linked_snomedCT_dataset.load_triple2coexist_num()
        filtered_triples = self.load_filtered_triples()
        filtered_triple2coexist_num = {}
        
        for triple in filtered_triples:
            coexist_num = original_triple2coexist_num[triple]
            filtered_triple2coexist_num[triple] = coexist_num
            
        return filtered_triple2coexist_num
        
    
    def __filter_relations_with_threshold(self, relation_triple_num_threshold = 100):
        """
        relation_data:
            'relation':relationname2relation[relationname],
            'relation name': relationname, 
            'triples_num': num, 
            'triples': list of triples_with_real_name
            the triples here is acyally the "triples_with_real_name"
        """
        relation_data_list = self.pubtator_linked_snomedCT_dataset.load_all_relation_data()
        
        filtered_relation_data_list = [single_relation_data for single_relation_data in relation_data_list if single_relation_data['triples_num'] > relation_triple_num_threshold]
        
        return filtered_relation_data_list
    
    def __filter_selected_relation_names(self, relation_data_list, relation_names_to_be_removed=[]):
        filtered_relation_data_list = []
        for relation_data in relation_data_list:
            if relation_data['relation name'] not in relation_names_to_be_removed:
                filtered_relation_data_list.append(relation_data)
        
        return filtered_relation_data_list
    
    
    def __filter_k_triples_for_each_relation(self, relation_data_list, k=5):
        random.seed(42)
        relation_name2k_triples_with_real_name = {}
        
        for relation_data in relation_data_list:
            relation_name = relation_data['relation name']
            triples_with_real_name = relation_data['triples']
            
            k_triples_with_real_name = random.sample(triples_with_real_name, min(len(triples_with_real_name), k))
            
            relation_name2k_triples_with_real_name[relation_name] = k_triples_with_real_name
            
            triples_with_real_name = [triple_with_real_name for triple_with_real_name in triples_with_real_name if triple_with_real_name not in k_triples_with_real_name]
            
            relation_data['triples'] = triples_with_real_name
        
        return relation_data_list, relation_name2k_triples_with_real_name
    
    
    
        
    
        
        
if __name__ == "__main__":
    pubtator_linked_to_snomedCT_dataset = PubtatorLinkedToSnomedCTDataset()
    # pubtator_linked_to_snomedCT_dataset.load_remaining_triples()
    # pubtator_linked_to_snomedCT_dataset.load_remaining_triples_with_real_names()
    # pubtator_linked_to_snomedCT_dataset.load_remaining_triple2triple_with_real_name()
    # pubtator_linked_to_snomedCT_dataset.load_all_relation_data()
    # pubtator_linked_to_snomedCT_dataset.load_remaining_relations()
    # pubtator_linked_to_snomedCT_dataset.load_relation2triples_distribution()
    # pubtator_linked_to_snomedCT_dataset.generate_relation_example()
    
    # pubtator_linked_to_snomedCT_dataset.draw_excell_based_on_relation_data_example()
    relation_names_to_be_removed=['Is a',
                                  'Pathological process (attribute)',
                                  'Procedure site',
                                  'Occurrence',
                                  'Procedure site - Direct (attribute)',
                                  'Method',
                                  'Has interpretation',
                                  'Laterality',
                                  'Has disposition (attribute)',
                                  'Component',
                                  'After',
                                  'Clinical course',
                                  'Course',
                                  'Finding method (attribute)',
                                  'Temporally follows',
                                  'Has focus',
                                  'Has intent',
                                  'Using device (attribute)',
                                  'Access',
                                  'Using',
                                  'Inheres in',
                                  'Extent',
                                  'Has realization (attribute)',
                                  'Severity',
                                  'Temporal context (attribute)',
                                  'Has device intended site (attribute)',
                                  'Plays role',
                                  'Property (attribute)',
                                  'Associated procedure',
                                  'Has specimen',
                                  'Using energy (attribute)',
                                  'During (attribute)',
                                  'Specimen source topography',
                                  'Communication with wound',
                                  'Procedure morphology (attribute)',
                                  'Has dose form (attribute)'
                                  ]
    
    filtered_pubtator_linked_to_snomedCT_dataset = FilterPubtatorLinkedToSnomedCTDataset(pubtator_linked_to_snomedCT_dataset,\
                                                                                         relation_names_to_be_removed=relation_names_to_be_removed,\
                                                                                         is_filter_process=True)
    
    filtered_pubtator_linked_to_snomedCT_dataset.load_filtered_relation_data_list()
    
    
    triples = filtered_pubtator_linked_to_snomedCT_dataset.load_filtered_triples()
    # 265666
    print(f"{len(triples)}")