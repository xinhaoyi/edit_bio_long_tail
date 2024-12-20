import sys
from typing import Any

import torch
from torch.utils.data import Dataset

sys.path.append("/nfs")
sys.path.append("/nfs/general")
sys.path.append("/nfs/long_tail")

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils.my_utils import load_json, random_state, save_json
from dsets.prompt_dataset import SnomedCTPromptDataset
from dataclasses import dataclass
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
                 add_demonstration_to_origin_prompt: bool=False,
                 rephrase_parrot = None):
        # self.tokenizer = tokenizer
        self.prompt_mode = prompt_mode
        self.prompt_dataset = SnomedCTPromptDataset(prompt_mode=prompt_mode)
        self.is_demonstration = is_demonstration
        self.is_rephrase = is_rephrase
        self.is_locality = is_locality
        self.is_portability = is_portability
        self.add_demonstration_to_origin_prompt = add_demonstration_to_origin_prompt
        self.rephrase_parrot = rephrase_parrot
        
    def __call__(self, batch):
        indexes = []
        prompts = []
        
        demonstrated_prompts = []
        
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
                demonstrated_prompts.append(prompt)
                
                # Only transform the prompt differently if the demonstration should not be added to the origin prompt
                if not self.add_demonstration_to_origin_prompt:
                    prompt = self.prompt_dataset.transform_triple_with_real_name_to_prompt(triple_with_real_name)
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
                rephrase_prompt = self.prompt_dataset.generate_rephrase_prompt(original_prompt=prompt, triple_with_real_name=triple_with_real_name, is_demonstration=self.is_demonstration)
                rephrase_prompts.append(rephrase_prompt)
            
            if self.is_locality:
                # ! todo
                locality_triple_with_real_name = data['locality_triple']
                locality_prompt, answer = self.prompt_dataset.generate_locality_prompt_and_answer(triple_with_real_name=locality_triple_with_real_name, is_demonstration=self.is_demonstration)
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
        
        
        ret_dict = {
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
        
        if self.is_demonstration and len(demonstrated_prompts) > 0:
            ret_dict.update(
                {'demonstrated_prompts': demonstrated_prompts}
            )
        
        return ret_dict

class SnomedCTKnowledgeTriplesProbingCollator:

    def __init__(self, prompt_mode: int = 0, is_demonstration: bool=False, is_rephrase: bool=False, is_locality: bool=False,):
        # self.tokenizer = tokenizer
        self.prompt_mode = prompt_mode
        self.prompt_dataset = SnomedCTPromptDataset(prompt_mode=prompt_mode)
        self.is_demonstration = is_demonstration
        self.is_rephrase = is_rephrase
        self.is_locality = is_locality

    def __call__(self, batch):
        
        indexes = []
        prompts = []
        labels = []
        
        subjects = []
        target_news = []
        rephrase_prompts = []
        
        locality_prompts = []
        locality_ans = []
        
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
            
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        } if self.is_locality else None
        
        return {
            'indexes': indexes,
            'prompts': prompts,
            'subjects': subjects,
            'target_news': target_news,
            'ground_truths': target_news,
            'rephrase_prompts': rephrase_prompts,
            'locality_inputs': locality_inputs,
            'batch_len': len(batch)
        }
    
    