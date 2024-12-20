import sys

# sys.path.append("/nfs")
# sys.path.append("/nfs/long_tail")
# sys.path.append("/nfs/general")

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle

from config import path_config


class KnowledgeGraphDataset:
    def __init__(self, save_data_path) -> None:
        self.save_data_path = save_data_path
        self.statistics = KnowledgeGraphDataset.KnowledgeGraphStatistics(self)
        
    def load_triples(self):
        triples_save_path = os.path.join(self.save_data_path, "triples.pickle")
        triples = pickle.load(open(triples_save_path, mode="rb"))
        return triples
    
    def load_entity2id(self):
        entity2id_save_path = os.path.join(self.save_data_path, "ent2id.pickle")
        ent2id = pickle.load(open(entity2id_save_path, mode="rb"))
        return ent2id
    
    def load_id2entityname(self):
        id2entityname_save_path = os.path.join(self.save_data_path, "id2entityname.pickle")
        id2entityname = pickle.load(open(id2entityname_save_path, mode="rb"))
        return id2entityname
    
    def load_relation2id(self):
        relation2id_save_path = os.path.join(self.save_data_path, "relation2id.pickle")
        relation2id = pickle.load(open(relation2id_save_path, mode="rb"))
        return relation2id
    
    def load_id2relationname(self):
        id2relationname_save_path = os.path.join(self.save_data_path, "id2relationname.pickle")
        id2relationname = pickle.load(open(id2relationname_save_path, mode="rb"))
        return id2relationname        
        
    
    class KnowledgeGraphStatistics:
        def __init__(self, umls_dataset: 'KnowledgeGraphDataset') -> None:
            self.umls_dataset = umls_dataset
        
        def num_of_triples(self):
            triples = self.umls_dataset.load_triples()
            return len(triples)
        
        def num_of_entites(self):
            entity2id = self.umls_dataset.load_entity2id()
            entities = [entity for entity in entity2id.keys()]
            
            return len(entities)
        
        def num_of_relations(self):
            triples = self.umls_dataset.load_triples()
            relation_set = set()
            for triple in triples:
                relation = triple[1]
                relation_set.add(relation)
            
            return len(relation_set)


default_save_snomedCT_data_path = path_config.SNOMEDCT_UMLS_SAVE_DATA_DIR

class SnomedCTDataset(KnowledgeGraphDataset):
    def __init__(self, save_snomedCT_data_path = default_save_snomedCT_data_path) -> None:
        super().__init__(save_data_path=save_snomedCT_data_path)

default_save_umsl_data_path = path_config.METATHESAURUS_UMLS_SAVE_DATA_DIR

class UMLSDataset(KnowledgeGraphDataset):
    def __init__(self, save_umls_data_path = default_save_umsl_data_path) -> None:
        super().__init__(save_data_path=save_umls_data_path)