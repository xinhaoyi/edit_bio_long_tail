import json
import pickle
import sys
from typing import Optional, Union, List, Tuple, Dict
import numpy as np
from time import time

# sys.path.append("/nfs")
# sys.path.append("/nfs/long_tail")
# sys.path.append("/nfs/general")

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from transformers import AutoModel, AutoTokenizer, AutoConfig, BioGptTokenizer, BioGptForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, GPT2Tokenizer, set_seed, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM

import logging

from ..util import HyperParams
from ..editors.editor import BaseEditor
from ..evaluate.evaluate import compute_edit_quality, compute_icl_edit_quality, compute_probing_quality

from ..util.globals import *
from ..util.hparams import HyperParams
from ..util.alg_dict import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)

def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class MyEditor(BaseEditor):
    """Base editor for all methods"""

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None or print('Error: hparams is None.')

        self.model_name = hparams.model_name
        if hparams.alg_name != "PROBING":
            self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")
        
        """
        model_name = "./hugging_cache/gpt-j-6B"
        """
        
        # from pdb import set_trace; set_trace()
        
        if type(self.model_name) is str:
            if 'biogpt' in self.model_name.lower():
                # self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, device_map='auto' if hparams.model_parallel else None)
                if 'biogpt-large' in self.model_name.lower():
                    self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt-large", device_map='auto' if hparams.model_parallel else None)
                    # self.tok = BioGptTokenizer.from_pretrained("microsoft/biogpt-large", padding_side='left')
                    self.tok = BioGptTokenizer.from_pretrained("microsoft/biogpt-large")
                else:
                    self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt", device_map='auto' if hparams.model_parallel else None)
                    self.tok = BioGptTokenizer.from_pretrained("microsoft/biogpt")
                    # self.tok = BioGptTokenizer.from_pretrained("microsoft/biogpt", padding_side='left')
            elif 'biomedlm' in self.model_name.lower():
                self.model = GPT2LMHeadModel.from_pretrained("stanford-crfm/BioMedLM")
                self.tok = GPT2Tokenizer.from_pretrained("stanford-crfm/BioMedLM")                
                self.tok.add_special_tokens({'pad_token': '[PAD]'})
                # self.tok.padding_side = "left"
                self.tok.unk_token_id = self.tok.pad_token_id
            elif  'gpt-neo'in self.model_name.lower():
                self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
                self.tok = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
                self.tok.pad_token = self.tok.eos_token
                # self.tok.padding_side = "left"
            elif 'gatortron-medium' in self.model_name.lower():
                self.model = AutoModel.from_pretrained('UFNLP/gatortron-medium')
                self.tok = AutoTokenizer.from_pretrained('UFNLP/gatortron-medium')
                # self.config=AutoConfig.from_pretrained('UFNLP/gatortron-medium')
            
            elif 'llama-2' in self.model_name.lower():
                hf_token = "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"
                self.model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=hf_token)
                self.tok = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=hf_token)
                self.tok.pad_token_id = self.tok.eos_token_id
            
            elif 'mistral' in self.model_name.lower():
                hf_token = "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"
                self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", token=hf_token)
                self.tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token=hf_token)
                self.tok.pad_token = self.tok.eos_token
            
            elif 'gpt-j' in self.model_name.lower():
                # 6B
                self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")
                self.tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
                # self.tok.add_special_tokens({'pad_token': '[PAD]'})
                self.tok.pad_token = self.tok.eos_token
            
            elif 'pmc_llama' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained("axiong/PMC_LLaMA_13B")
                self.tok = AutoTokenizer.from_pretrained("axiong/PMC_LLaMA_13B")
                self.tok.pad_token_id = self.tok.eos_token_id
                
            else:
                raise NotImplementedError
            
            # if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, LlamaTokenizer) or isinstance(self.tok, BioGptTokenizer))  and (hparams.alg_name not in ['ROME', 'MEMIT']):
            #     LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
            #     self.tok.padding_side = 'left'
            if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, BioGptTokenizer)  or isinstance(self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and ('mistral' in self.model_name.lower() or 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
            
        else:
            self.model, self.tok = self.model_name
        # device_map = {
        #     0: [_ for _ in range(0, 16)],
        #     1: [_ for _ in range(16, 32)],
        #     2: [_ for _ in range(32, 48)]
        # }
        # self.model.parallelize(device_map=device_map)
        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams
        
    
    
    def probe(self,
                      prompts: Union[str, List[str]],
                      target_new: Union[str, List[str]],
                      ground_truth: Optional[Union[str, List[str]]] = None,
                      rephrase_prompts: Optional[Union[str, List[str]]] = None,
                      locality_inputs:  Optional[Dict] = None,
                      verbose=True,
                      **kwargs
                      ) -> List[Dict[str, Dict[str, Union[float, Dict[str, float]]]]]:
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        # test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        # if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
        #     self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        # assert (locality_prompts is None and locality_ground_truth is None) or \
        #        (isinstance(locality_prompts, str) and isinstance(locality_ground_truth, str)) or \
        #        len(locality_prompts) == len(locality_ground_truth) or print('Error in locality Input.')

        requests = self._prepare_probe_requests(prompts, target_new, ground_truth, rephrase_prompts, locality_inputs, **kwargs)
        
        temperature = kwargs['temperature'] if 'temperature' in kwargs.keys() else 1.0
        
        is_test_dif_temperature = kwargs['is_test_dif_temperature'] if 'is_test_dif_temperature' in kwargs.keys() else False
        
        is_test_dif_gold_distributions = kwargs['is_test_dif_gold_distributions'] if 'is_test_dif_gold_distributions' in kwargs.keys() else False
        
        is_test_multi_ans_rephrase = kwargs['is_test_multi_ans_rephrase'] if 'is_test_multi_ans_rephrase' in kwargs.keys() else False
        
        
        all_metrics = []
        
        # from pdb import set_trace; set_trace()
        for i, request in enumerate(requests):
            metrics = {
                "pre": compute_probing_quality(
                    model=self.model, 
                    model_name=self.model_name, 
                    hparams=self.hparams, 
                    tok=self.tok, 
                    record=request,
                    device=self.hparams.device,
                    temperature=temperature,
                    is_test_dif_temperature=is_test_dif_temperature,
                    is_test_dif_gold_distributions=is_test_dif_gold_distributions,
                    is_test_multi_ans_rephrase=is_test_multi_ans_rephrase,
                    )
            }
            all_metrics.append(metrics)
            if verbose:
                LOG.info(
                    f"{i} Probing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                )
        
        return all_metrics
    
    
    def pure_edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             keep_original_weight=False,
             verbose=True,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]
        
        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1
        
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]
            
        requests = self._prepare_requests(prompts=prompts, 
                                          target_new=target_new, 
                                          ground_truth=ground_truth,
                                          **kwargs)
        
        
        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1 or \
                      print(f'Single Edit, pls set the batch_size to 1....')
        
        empty_metrics = []
        
        for i, request in enumerate(requests):
            start = time()
            if self.alg_name == 'IKE':
                """
                IKE Editing Method
                """
                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds(For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n"
                    )
                # take care! the ike pure edit will return a extra icl_examples
                return empty_metrics, edited_model, weights_copy, icl_examples
            
            else:
                """
                Other Editing Methods
                """
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                
                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n"
                    )
                
                return empty_metrics, edited_model, weights_copy
    
    
    def _prepare_probe_requests(self,
                          prompts: Union[str, List[str]],
                          target_new: Union[str, List[str]],
                          ground_truth: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          locality_inputs:  Optional[Dict] = None,
                          **kwargs
                          ) -> List[
                              Dict[
                                  str, Union[
                                      str, Dict[
                                          str, List[Dict[str, str]]
                                          ]
                                      ]
                                  ]
                              ]:

        requests: List[Dict[str, Union[str, Dict[str, List[Dict[str, str]]]]]]\
        = [{
            'prompt': prompt,
            'target_new': target_new_,
            'ground_truth': ground_truth_,
        }
        for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
        ]

        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_ or print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': rephrase_prompts[i],
                    }
                )
        
        # from pdb import set_trace; set_trace()
                
        if locality_inputs is not None:
            for locality_key in locality_inputs.keys():
                if isinstance(locality_inputs[locality_key]['prompt'], str):
                    locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                    locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
                assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
                == len(requests) or print('One Edit instance needs one locality input.....')

                for i, request in enumerate(requests):
                    if 'locality' not in request:
                        request.update(
                            {'locality':{}}
                        )
                    request['locality'].update(
                        {
                            locality_key: {
                                f'prompt': locality_inputs[locality_key]['prompt'][i],
                                f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                            }
                        }
                    )
        
        

        return requests
    
    
    
    
    
    
        
                
            