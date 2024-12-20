import json
import os
import pickle
import time
import numpy as np
import torch

from tqdm import tqdm, trange
from abc import ABC, abstractmethod
from collections import deque

from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist 
from torch.utils.data.distributed import DistributedSampler

def load_pickle(file_path):
    """
    Load the pickle file.

    Parameters: file_path (str): path to the pickle file.
        file_path (str): path of the pickle file.

    Returns: obj
        obj: Deserialised object.
    """
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except pickle.UnpicklingError:
        print(f"Error: File at {file_path} is not a valid pickle file")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def save_pickle(obj, file_path):
    """
    Save the object to the pickle file and add exception handling.

    Parameters.
        obj: the object to save.
        file_path (str): Path to the file to save.
    """
    try:
        with open(file_path, "wb") as f:  # Write binary file using 'wb' mode
            pickle.dump(obj, f)
        # print(f"Data saved to {file_path}")
    except FileNotFoundError:
        print(f"Error: Directory for file path '{file_path}' does not exist.")
    except PermissionError:
        print(f"Error: No permission to write to file path '{file_path}'.")
    except pickle.PickleError:
        print("Error: An issue occurred while pickling the object.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



# def load_json(path, type="json"):
#     assert type in ["json", "jsonl"] # only support json or jsonl format
#     with tqdm(total=1, desc="Loading JSON ...") as pbar:
#         if type == "json":
#             outputs = json.loads(open(path, "r", encoding="utf-8").read())
#         elif type == "jsonl":
#             outputs = []
#             with open(path, "r", encoding="utf-8") as fin:
#                 for line in fin:
#                     outputs.append(json.loads(line))
#         else:
#             outputs = []
        
#         # 标记保存完成
#         pbar.update(1)
        
#     return outputs

def load_json(path, type="json"):
    assert type in ["json", "jsonl"] # only support json or jsonl format
    if type == "json":
        outputs = json.loads(open(path, "r", encoding="utf-8").read())
    elif type == "jsonl":
        outputs = []
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                outputs.append(json.loads(line))
    else:
        outputs = []
    return outputs


# def save_json(data, path, type="json", use_indent=False):
#     assert type in ["json", "jsonl"] # only support json or jsonl format
#     with tqdm(total=1, desc="Saving to JSON ...") as pbar:
#         start_time = time.time()  # 记录开始时间
#         if type == "json":
#             with open(path, "w", encoding="utf-8") as fout:
#                 if use_indent:
#                     fout.write(json.dumps(data, indent=4))
#                 else:
#                     fout.write(json.dumps(data))

#         elif type == "jsonl":
#             with open(path, "w", encoding="utf-8") as fout:
#                 for item in data:
#                     fout.write("{}\n".format(json.dumps(item)))
                    
#         end_time = time.time()  # 记录结束时间
#         elapsed_time = end_time - start_time  # 计算花费的时间

#         # 使用 tqdm.set_postfix 更新进度条附加信息，包括已用时间
#         pbar.set_postfix({"Time": f"{elapsed_time:.2f} seconds"})
        
#         # 标记保存完成
#         pbar.update(1)

#     return path

def save_json(data, path, type="json", use_indent=False):
    assert type in ["json", "jsonl"] # only support json or jsonl format
    if type == "json":
        with open(path, "w", encoding="utf-8") as fout:
            if use_indent:
                fout.write(json.dumps(data, indent=4))
            else:
                fout.write(json.dumps(data))
    elif type == "jsonl":
        with open(path, "w", encoding="utf-8") as fout:
            for item in data:
                fout.write("{}\n".format(json.dumps(item)))
            
    return path


def count_lines_in_file(file_path):
    """
    计算文件中的行数。

    Args:
        file_path (str): 文件路径。

    Returns:
        int: 文件中的行数。
    """
    line_count = 0
    with open(file_path, 'r') as file:
        for _ in file:
            line_count += 1
    return line_count


def check_file_exists(file_path, is_desc = False):
    """
    检查文件是否存在。

    Args:
        file_path (str): 要检查的文件路径。

    Returns:
        bool: 如果文件存在，返回 True；否则返回 False。
    """
    
    if os.path.exists(file_path):
        if is_desc:
            print(f"File {file_path} exists")
        return True
    else:
        if is_desc:
            print(f"File {file_path} doesn't exist")
        return False
    
    
def count_entity_distribution_with_dif_num_papers(mentions2PMIDs, is_print=False):
    n_PMIDs2entities = dict()
    for mention, PMIDs_set in mentions2PMIDs.items():
        n_PMIDs = len(PMIDs_set)
        if n_PMIDs not in n_PMIDs2entities.keys():
            n_PMIDs2entities[n_PMIDs] = list()
        n_PMIDs2entities[n_PMIDs].append(mention)
    num_dict = {'0_2': 0, '2_5': 0, '5_10': 0,'10_20': 0, '20_30': 0, '30_40': 0, '40_50': 0, '50_60': 0, '60_70': 0, '70_80': 0, '80_90': 0, '90_100': 0, '100_500': 0, '500_1000': 0, '1000_2000': 0, '2000_5000': 0, 'more_5000': 0}
    n_PMIDs2entities_items = n_PMIDs2entities.items()
    total_length_n_PMIDs2entities_items = len(n_PMIDs2entities_items)
    
    num_dict = {
        '0_2': 0,
        '2_5': 0,
        '5_10': 0,
        '10_20': 0,
        '20_30': 0,
        '30_40': 0,
        '40_50': 0,
        '50_60': 0,
        '60_70': 0,
        '70_80': 0,
        '80_90': 0,
        '90_100': 0,
        '100_500': 0,
        '500_1000': 0,
        '1000_2000': 0,
        '2000_5000': 0,
        'more_5000': 0
    }
    
    n_PMIDs_values = [0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000, 2000, 5000]
    
    for n_PMIDs, entities in tqdm(n_PMIDs2entities_items, desc=f"Processing the mentions2PMIDs dictionary ...", total=total_length_n_PMIDs2entities_items):
        n_entities = len(entities)
        for i in range(len(n_PMIDs_values) - 1):
            lower_limit = n_PMIDs_values[i]
            upper_limit = n_PMIDs_values[i + 1]
            
            key = f"{lower_limit}_{upper_limit}"
            
            if lower_limit <= n_PMIDs < upper_limit:
                num_dict[key] += n_entities
        
        # 处理 'more_5000' 范围
        if n_PMIDs >= 5000:
            num_dict['more_5000'] += n_entities
        
    if is_print:
        for key, value in num_dict.items():
                print(f"The number of mentions linking to {key} papers: {value}")
        
    return num_dict
    
def calculate_mention2PMIDs_statistics(mention2PMIDs: dict, file_name: str, is_print = True):
    print(f"statistic the dictionary {file_name}_mention2PMIDs.pickle ...")
    data_process_path = '/nfs/long_tail/datasets/pubtator/process'
    PMIDs_set = set()
    for PMIDs in mention2PMIDs.values():
        PMIDs_set.update(PMIDs)
        
    num_mentions = len(mention2PMIDs)
    num_papers = len(PMIDs_set)
    num_dict = count_entity_distribution_with_dif_num_papers(mention2PMIDs)
    
    if is_print:
        print(f"--------The Statistics information of the dictionary {file_name}_mention2PMIDs.pickle--------")
        print(f"num of the mentions: {num_mentions}")
        print(f"num of the papers(PMIDs): {num_papers}")
        for key, value in num_dict.items():
            print(f"The number of mentions linking to {key} papers: {value}")
    
    return {'num_mentions': num_mentions,
            'num_papers': num_papers,
            'num_dict': num_dict
            }
    
    
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
    

def parse_knowledge_triples_data(knowledge_triples_data, is_set_N_N_relationship_type = False):
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
        
        for relation, knowledge_triples_data in relation2knowledge_triples_data.items():
            head_entity2count = {}
            tail_entity2_count = {}
            for knowledge_triple_data in knowledge_triples_data:
                head_entity = knowledge_triple_data['triple'][0]
                tail_entity = knowledge_triple_data['triple'][2]
                head_entity2count[head_entity] = head_entity2count.get(head_entity, 0) + 1
                tail_entity2_count[tail_entity] = tail_entity2_count.get(tail_entity, 0) + 1
            if not is_set_N_N_relationship_type:
                for single_knowledge_triple_data in knowledge_triples_data:
                    head_entity = single_knowledge_triple_data['triple'][0]
                    tail_entity = single_knowledge_triple_data['triple'][2]
                    
                    if head_entity2count[head_entity] == 1 and tail_entity2_count[tail_entity] == 1:
                        relation2knowledge_triples_data_1_to_1.setdefault(relation, []).append(single_knowledge_triple_data)
                    elif head_entity2count[head_entity] > 1 and tail_entity2_count[tail_entity] == 1:
                        relation2knowledge_triples_data_1_to_N.setdefault(relation, []).append(single_knowledge_triple_data)
                    elif head_entity2count[head_entity] == 1 and tail_entity2_count[tail_entity] > 1:
                        relation2knowledge_triples_data_N_to_1.setdefault(relation, []).append(single_knowledge_triple_data)
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

def get_dataloader(local_rank: int, dataset: Dataset, batch_size: int, shuffle: bool, collate_fn=None, drop_last: bool=False):
    
    if local_rank >= 0:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=local_rank, drop_last=drop_last, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=False, num_workers=0, sampler=sampler, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, collate_fn=collate_fn)

    return dataloader

def get_avg_scores_dict_from_scores_dict_list(scores_dict_list):
    if scores_dict_list is None or len(scores_dict_list) == 0:
        return {}
    average_scores = {}
    # 动态获取所有可能的键
    all_keys = set(key for score_dict in scores_dict_list for key in score_dict.keys())
    # 初始化累加器字典
    sums = {key: 0 for key in all_keys}
    
    # 遍历每个列表中的分数字典
    for score_dict in scores_dict_list:
        for key in all_keys:
            sums[key] += sum(score_dict.get(key, [0]))  # 使用sum处理可能的列表值
    
    # 计算平均值并存储
    average_scores = {key: total / len(scores_dict_list) for key, total in sums.items()}
    
    return average_scores
    
class MultiFileIterator:
    def __init__(self):
        """ 初始化迭代器。

        :param filepaths: 文件路径列表。
        """
        self.filepaths = []
        self.current_file = None
        self.file_index = 0
        
    def add_file(self, filepath):
        """ 添加新的文件路径。"""
        self.filepaths.append(filepath)
        
    def add_file_list(self, filepath_list):
        for file_path in filepath_list:
            self.add_file(filepath=file_path)

    def __iter__(self):
        return self

    def __next__(self):
        """ 返回下一个文件的内容。"""
        # 如果当前文件已经打开，先关闭它
        # if self.current_file:
        #     self.current_file.close()
        #     self.current_file = None

        # 如果所有文件都已处理，停止迭代
        if self.file_index >= len(self.filepaths):
            raise StopIteration

        # 打开下一个文件
        # self.current_file = open(self.filepaths[self.file_index], 'r')
        data = np.load(self.filepaths[self.file_index])
        self.file_index += 1

        return data

    # def close(self):
    #     """ 关闭当前打开的文件（如果有的话）。"""
    #     if self.current_file:
    #         self.current_file.close()
    #         self.current_file = None



class LargeDataProcessor:
    def __init__(self, save_file_type, batch_size) -> None:
        self.save_file_type = save_file_type
        self.batch_size = batch_size
        self.multi_file_iterator = MultiFileIterator()
    
    @abstractmethod
    def test_abstract(self):
        pass

class LargeEmbeddingProcessor(LargeDataProcessor):
    def __init__(self, save_file_type, batch_size, save_path: str, save_file_name: str) -> None:
        super().__init__(save_file_type, batch_size)
        self.__save_path = save_path
        self.__save_file_name = save_file_name
        self.__embedding_pool: deque = deque([])
        self.__n_embeddings_in_pool: int = 0
        self.__n_total_file = 0
        self.__file_index = 0
        self.__save_path_list = []
        
    def __append_data_to_pool(self, embeddings):
        # print(embeddings)
        for embedding in embeddings:
            # print(embedding)
            self.__embedding_pool.append(embedding.tolist())
        self.__n_embeddings_in_pool += len(embeddings)
        
    def __pop_batch_data_from_pool(self, batch_size):
        batch_data= [self.__embedding_pool.pop() for _ in range(min(batch_size, self.__n_embeddings_in_pool))]
        # 更新pooli里的embeddings数量
        self.__n_embeddings_in_pool -= min(batch_size, self.__n_embeddings_in_pool)
        # print(batch_data)
        return batch_data
    
    def __clear_pool(self):
        self.__embedding_pool.clear()
        self.__n_embeddings_in_pool = 0
        
    def __del_files(self):
        pass
        
    def __forward_file_index(self):
        save_file_base_name, save_file_extension = os.path.splitext(self.__save_file_name)
        save_batch_file_name = f"{save_file_base_name}_part_{self.__file_index}{save_file_extension}"
        save_batch_file_path = os.path.join(self.__save_path, save_batch_file_name)
        
        self.__file_index += 1
        
        return save_batch_file_name, save_batch_file_path
    
# ========================================================
#                   BIGIN SAVE DATA PART
# ========================================================
        
    def append_data_with_auto_save(self, new_embeddings: list):
        
        def valid(input):
            return isinstance(input, torch.Tensor) or isinstance(input, np.ndarray)
        
        def convert(input):
            if isinstance(new_embeddings, torch.Tensor):
                input = input.cpu().detach().numpy()
            return input
        
        if not isinstance(new_embeddings, list):
            new_embeddings = [new_embeddings]
        
        if not all([valid(item) for item in new_embeddings]):
            raise TypeError("Embeddings must be a NumPy array or a PyTorch tensor.")
        
        new_embeddings = [convert(item) for item in new_embeddings]
        
        # ! delete
        # from pdb import set_trace; set_trace()
        for one_embedding in new_embeddings:
            self.__append_data_to_pool(one_embedding)
        
        # ! delete
        # from pdb import set_trace; set_trace()
        
        while self.__n_embeddings_in_pool >= self.batch_size:
            self.__save_one_batch_data_to_file()
            
    def flush(self):
        self.__save_all_data_to_file()
        save_file_base_name, save_file_extension = os.path.splitext(self.__save_file_name)
        index_list = [i for i in range(self.__file_index)]
        save_series_first_name = f"{save_file_base_name}_part_{index_list[0]}{save_file_extension}"
        save_series_last_name = f"{save_file_base_name}_part_{index_list[-1]}{save_file_extension}"
        save_series_file_path = f"{os.path.join(self.__save_path, save_series_first_name)} ----------\n---------- {os.path.join(self.__save_path, save_series_last_name)}"
        
        print(f"Successfully Saving All Embeddings In {save_series_file_path} !!! ")
            
            
    # def count_n_file_needed(self):
    #     return (self.__n_embeddings_in_pool - 1)//self.batch_size + 1

    
    def __save_one_batch_data_to_file(self):
        """
        This method will save a batch of data within pool to a static file.
        If the data within pool is less than a batch, it will automatically save the remaining data into the file.
        """
        batch_embedding_list = self.__pop_batch_data_from_pool(batch_size=self.batch_size)
        
        # batch_embeddings = np.concatenate(batch_embedding_list, axis=0)
        
        save_batch_file_name, save_batch_file_path = self.__forward_file_index()
        
        # ! delete
        # from pdb import set_trace; set_trace()
        
        print(f"saving batch embeddings to {save_batch_file_path} ... With {self.__n_embeddings_in_pool} lines of data remaining")
        np.save(save_batch_file_path, batch_embedding_list)
        self.__save_path_list.append(save_batch_file_path)
        self.multi_file_iterator.add_file(save_batch_file_path)
        
    def __save_all_data_to_file(self):
        batch_num: int = (self.__n_embeddings_in_pool - 1)//self.batch_size + 1
        for i in trange(batch_num):
            self.__save_one_batch_data_to_file()
            
            
# ========================================================
#                   END SAVE DATA PART
# ========================================================
    
        



if __name__ == "__main__":
    processor = LargeEmbeddingProcessor(save_file_type='npy', batch_size=3, save_path='/nfs/long_tail/datasets/usml/SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z/', save_file_name='test.npy')
    # for i in range(10):
    emb_list = [torch.randn(10,3), torch.randn(9,3), torch.randn(10,3)]
    # print(emb_list)
    
    # ! delete
    # from pdb import set_trace; set_trace()
    
    processor.append_data_with_auto_save(emb_list)
    processor.flush()
    
    for file_content in processor.multi_file_iterator:
        print(file_content)
    
    
    
    
    
            
            
    

            

