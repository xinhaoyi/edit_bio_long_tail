from argparse import Namespace
from typing import Dict, List
from edit.util.hparams import HyperParams
from dataclasses import dataclass


def average_metric_dicts(dict_list: List[Dict], forbid_keys: List[str] = None):
    if not dict_list:
        return {}
    
    if forbid_keys is None:
        forbid_keys = []
    
    def recurse_sum_dicts(dicts, result=None):
        """递归求和字典中的数值"""
        if result is None:
            result = {}
        for key, value in dicts[0].items():
            if key in forbid_keys:
                continue
            if isinstance(value, dict):
                # 如果当前项是字典，递归处理
                result[key] = recurse_sum_dicts([d[key] for d in dicts], result.get(key, {}))
            elif isinstance(value, list):
                # 如果是列表，对应位置求和
                result[key] = [sum(items) for items in zip(*[d[key] for d in dicts])]
            else:
                # 如果是数值，计算和
                result[key] = sum(d[key] for d in dicts)
        return result

    def divide_dict(nested_dict, divisor):
        """递归除以某个数，用于求平均值"""
        for key, value in nested_dict.items():
            if key in forbid_keys:
                continue
            if isinstance(value, dict):
                divide_dict(value, divisor)
            elif isinstance(value, list):
                nested_dict[key] = [v / divisor for v in value]
            else:
                nested_dict[key] /= divisor
    
    # 使用递归求和
    sum_dict = recurse_sum_dicts(dict_list)
    # 求平均值
    divide_dict(sum_dict, len(dict_list))
    return sum_dict




def get_probe_hyperparams(args: Namespace):
    """
        hparams_dir = './hparams/ROME'
    """
    @dataclass
    class ProbingHyperParams(HyperParams):
        model_name: str = 'GPT-2' # default
        device: int = 0
        # Defaults
        alg_name: str = "PROBING"
        batch_size: int = 64
        max_length: int = 500
        model_parallel: bool = False
    
    hparams = ProbingHyperParams()
    
    def model_name2model_path(model_name):
        transfer_dict = {
            "GPT-2": "openai-community/gpt2-xl",
            "GPT-Neo": "EleutherAI/gpt-neo-2.7B",
            "GPT-J": "EleutherAI/gpt-j-6b",
            "BioMedLM": "stanford-crfm/BioMedLM",
            "BioGPT-Large": "microsoft/biogpt-large",
        }
        if model_name in transfer_dict:
            return transfer_dict[model_name]
        else:
            raise Exception(f"{model_name} is not in {transfer_dict}, we currently can't support this model.")
    
    # from pdb import set_trace; set_trace()
    hparams = overwrite_hparams_from_args(hparams=hparams, args=args)
    
    # hparams_dir = ./hparams/ROME/gpt2-xl
    
    # hparams.model_name = model_name2model_path(hparams.model_name)
    
    return hparams

def overwrite_hparams_from_args(hparams: HyperParams, args: Namespace) -> HyperParams:
    # 获取 HyperParams 类的所有字段名
    valid_keys = set(vars(hparams).keys())
    
    # 根据 args 更新 hparams 中的字段，只有当键在 HyperParams 中定义时才更新
    for key, value in vars(args).items():
        if key in valid_keys and value is not None:
            setattr(hparams, key, value)  # 更新符合条件的值
    
    return hparams

