import pickle
import sys
from typing import Dict, List, Union

import numpy as np

# sys.path.append("/nfs")
# sys.path.append("/nfs/long_tail")
# sys.path.append("/nfs/general")

from config import PathConfig

# sys.path.append(PathConfig.BASE_DIR)


import os

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from tqdm import tqdm

import argparse
# from typing import Any
from dsets.knowledge_triples_dataset import SnomedCTKnowledgeTriplesDataset
# from dsets.knowledge_graph_dataset import SnomedCTDataset
from utils.collators import SnomedCTKnowledgeTriplesProbingCollator

import torch
# import pandas as pd
# from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, BioGptTokenizer, BioGptForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, GPT2Tokenizer, set_seed

from entity_linking_preprocess.snomedCT_entity_linking import sapbert_entity_linking

from utils.my_utils import get_dataloader
from utils.my_utils import save_json,get_avg_scores_dict_from_scores_dict_list
from utils.my_evaluation import exact_math_score_extend

from edit.editors.my_editor import MyEditor
from utils.knowledge_utils import average_metric_dicts, get_probe_hyperparams 


from config import PathConfig


root_path = str(PathConfig.BASE_DIR)

# model_selector = {
#     'BioGPT': {
#         'tokenizer': BioGptTokenizer.from_pretrained("microsoft/biogpt", padding_side='left'),
#         'model': BioGptForCausalLM.from_pretrained("microsoft/biogpt")
#     },
#     'BioGPT-Large': {
#         'tokenizer': BioGptTokenizer.from_pretrained("microsoft/biogpt-large", padding_side='left'),
#         'model': BioGptForCausalLM.from_pretrained("microsoft/biogpt-large")
#     },
#     'BioMedLM': {
#         'tokenizer': GPT2Tokenizer.from_pretrained("stanford-crfm/BioMedLM"),
#         'model': GPT2LMHeadModel.from_pretrained("stanford-crfm/BioMedLM")
#     },
#     'GPT-Neo':{
#         'tokenizer': GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B"),
#         'model': GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
#     },
#     'Gatortron-Medium': {
#         'tokenizer': AutoTokenizer.from_pretrained('UFNLP/gatortron-medium'),
#         'model': AutoModel.from_pretrained('UFNLP/gatortron-medium')
#     }
# }

# def select_model_and_tokenizer(model_name: str):
#     if model_name.lower().strip() == 'biogpt':
#         tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt", padding_side='left')
#         model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
#     elif model_name.lower().strip() == 'biogpt-large':
#         tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt-large", padding_side='left')
#         model = BioGptForCausalLM.from_pretrained("microsoft/biogpt-large")
#     elif model_name.lower().strip() == 'biomedlm':
#         tokenizer = GPT2Tokenizer.from_pretrained("stanford-crfm/BioMedLM")
#         model = GPT2LMHeadModel.from_pretrained("stanford-crfm/BioMedLM")
#     elif model_name.lower().strip() == 'gpt-neo':
#         tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
#         model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    
#     return model, tokenizer
    
    



def setup_parser():
    # nargs='+' make it available to receive one or more values (a list)
    parser = argparse.ArgumentParser(description="Run sweep function with command line arguments")
    parser.add_argument("--frequency_type", nargs='+', required=False, default= ['very_high', 'high', 'medium', 'long_tail'], help="List of frequency types")
    parser.add_argument("--model_name", type=str, required=False, default='BioGPT-Large')
    parser.add_argument("--prompt_mode", nargs='+', type=int, required=False, default=[0,1,2,3,4])
    parser.add_argument("--use_demonstration", action="store_true")
    # parser.add_argument("--use_entity_linking_eval", action="store_true")
    parser.add_argument("--use_sampling", action="store_true")
    parser.add_argument("--use_eq_sampling", action="store_true")
    parser.add_argument("--use_stratified_sampling", action="store_true")
    parser.add_argument("--cal_NN_score", action="store_true")
    
    parser.add_argument("--verbose", action="store_true")
    
    parser.add_argument("--save_triple_level_res", action="store_true")
    parser.add_argument("--metrics_save_dir", default='./', type=str)
    
    # cal_NN_score
    return parser.parse_args()
    

# def get_all_entities():
#     sn_save_process_data_path = "/nfs/long_tail/data/umls/SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z/process"
#     id2entityname = pickle.load(open(os.path.join(sn_save_process_data_path, "id2entityname.pickle"), mode="rb"))
#     entity_list = [id2entityname[i] for i in range(len(id2entityname))]
#     return entity_list



# def evaluation(results, labels, batch_size):
#     # scores = 0
#     scores_list = []
#     for single_result, label in zip(results, labels):
#         score = exact_math_score_extend(single_result, label)
#         scores_list.append(score)
#         # scores += score
    
#     return scores_list, batch_size

# def evaluation_ACC(ans_tokens, label_tokens):
#     if isinstance(ans_tokens[0], list):
#         res = []
#         for ans,label in zip(ans_tokens,label_tokens):
#             temp_acc = np.mean(np.equal(ans, label))
#             if np.isnan(temp_acc):
#                 continue
#             res.append(temp_acc)
#         return res
#     else:
#         return [np.mean(np.equal(ans_tokens, label_tokens))]



# def evaluation_with_entity_linking(results, labels, batch_size, all_entities):
#     scores = 0
#     for single_result, label in zip(results, labels):
        
#         result = sapbert_entity_linking([single_result])
        
#         id = int(result[0][0][0])
        
#         nearset_single_result = all_entities[id]
        
#         score = exact_math_score_extend(nearset_single_result, label)
#         scores += score
    
#     return scores / batch_size


# unused
def slice_list(matrix,start_indices,left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0]-1:-1]
        else:
            return matrix[start_indices[0]:]

# unused
def get_position_ids(attention_mask):
    
    assert len(attention_mask.shape) == 2 # attention mask 必须是二维的
    batch_size, seq_len = attention_mask.shape
    sum_attention_mask = torch.sum(attention_mask, dim=0)
    if sum_attention_mask[0] == batch_size:
        # 右 padding
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    elif sum_attention_mask[-1] == batch_size:
        # 左 padding 
        position_ids_list = []
        for i in range(batch_size):
            num_tokens = attention_mask[i].sum()
            position_ids_list.append([0] * (seq_len-num_tokens) + list(range(num_tokens)))
        position_ids = torch.tensor(position_ids_list, dtype=torch.long)
    else:
        print("Could not detect left padding or right padding, setting the position_ids to None!")
        position_ids = None
    
    return position_ids

# unused
def generate_answers_tokens_and_labels_tokens(model, tokenizer, prompts, targets, device, model_name):
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    
    if isinstance(targets[0], list):
        targets = [item for sublist in targets for item in sublist]
    
    # from pdb import set_trace; set_trace()
    prompt_targets = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tokenizer.encode(_)) for _ in prompt_targets]) + 1
    prompt_target_tok = tokenizer(
        prompt_targets,
        padding=True,
        truncation=True,
        max_length=max(100, max_prompt_len),
        return_tensors="pt",
    )
     
    # BioGPT-Large will modify the position_ids automatically
    if model_name in ['BioMedLM', 'GPT-Neo']:
        prompt_target_tok["position_ids"] = get_position_ids(prompt_target_tok["attention_mask"])
    
    prompt_tok = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(100, max_prompt_len),
        return_tensors="pt",
    )
    # from pdb import set_trace; set_trace()
    prompt_target_tok = {k: v.to(device) if torch.is_tensor(v) else v for k, v in prompt_target_tok.items()}
    prompt_tok = {k: v.to(device) if torch.is_tensor(v) else v for k, v in prompt_tok.items()}
    
    num_prompt_toks = [int((i != tokenizer.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tokenizer.pad_token_id).sum()) for i in prompt_target_tok['input_ids']]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    with torch.no_grad():
        # outputs = model.generate(**prompt_target_tok)
        # print(outputs)
        # from pdb import set_trace; set_trace()
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers,prompt_len,left=True)
        labels = slice_list(labels,prompt_len,left=False)
        
    return answers, labels


def main(frequency_type: str, 
         model_name: str, 
         prompt_mode: int, 
         is_demonstration: bool, 
         metrics_save_dir: str,
         use_sampling: bool = True, 
         use_eq_sampling: bool = False, 
         use_stratified_sampling: bool = False,
         is_example: bool =False, 
         save_triple_level_res: bool = False,
         cal_NN_score: bool = False
         ):
    print("\n")
    print("\n")
    print(f"####### {frequency_type} #######")
    print(f"####### {model_name} #######")
    print(f"####### Prompt Mode = {prompt_mode} #######")
    if is_demonstration:
        print(f"####### Use Demonstration #######")
    else:
        print(f"####### Don't Use Demonstration #######")
    
    # if use_entity_linking_eval:
    #     print("####### Contain Entity Linking Evaluation #######")
    # else:
    #     print("####### Don't Contain Entity Linking Evaluation #######")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset & collator
    if use_sampling == use_eq_sampling == True:
        use_eq_sampling = False
    
    frequency_dict = {
                'long_tail': (0,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None)
            }
    
    
    if use_sampling:
        dataset = SnomedCTKnowledgeTriplesDataset(frequency_type=frequency_type, frequency_dict=frequency_dict, save_data_file_name='clinic_knowledge_triples_dc-sampled.json', is_sample=True)
    elif use_eq_sampling:
        dataset = SnomedCTKnowledgeTriplesDataset(frequency_type=frequency_type, frequency_dict=frequency_dict, save_data_file_name='clinic_knowledge_triples_eq-sampled.json', is_sample=True)
    elif use_stratified_sampling:
        dataset = SnomedCTKnowledgeTriplesDataset(frequency_type=frequency_type, frequency_dict=frequency_dict, save_data_file_name='clinic_knowledge_triples_stratified_without_1_N_sampled.json', is_sample=True)
    else:
        dataset = SnomedCTKnowledgeTriplesDataset(frequency_type=frequency_type, frequency_dict=frequency_dict, save_data_file_name='clinic_knowledge_triples.json', is_sample=True)
    
    # model, tokenizer = select_model_and_tokenizer(model_name)
    
    # tokenizer = model_selector[model_name]['tokenizer']
    
    # from pdb import set_trace; set_trace()
    
    # if model_name == 'BioMedLM':
    #     # tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     tokenizer.padding_side = "left"
        
    # if model_name =='GPT-Neo':
    #     # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.padding_side = "left"
    
    
    collator = SnomedCTKnowledgeTriplesProbingCollator(prompt_mode=prompt_mode, is_demonstration=is_demonstration, is_rephrase=True, is_locality=True)
    dataloader = get_dataloader(-1, dataset, batch_size=8, shuffle=False, collate_fn=collator)
    
    # args.model_name = args.model_name[0]
    
    hparams = get_probe_hyperparams(args=args)
    
    editor = MyEditor.from_hparams(hparams)
    
    if save_triple_level_res:
        index2score = {}
    
    if cal_NN_score:
        metrics_1_1_list: List[Dict] = []
        metrics_1_N_list: List[Dict] = []
        metrics_N_1_list: List[Dict] = []
        metrics_N_N_list: List[Dict] = []

    # load model 
    # model = model_selector[model_name]['model']
    # model.to(device)
    # model.eval()
    # print(f"The number of triples is {dataset.get_n_triples()}")
    # print(f"The number of relations is {dataset.get_n_relations()}")

    # evaluate 
    with torch.no_grad():
        # all_outputs = []
        overall_acc_res = []
        overall_acc = 0
        # overall_EM_scores = 0
        # overall_EM_scores_entity_linking = 0
        
        total_data_len = 0
        
        # if use_entity_linking_eval:
        #     all_entities = get_all_entities()
        
        all_metrics: List[Dict] = []
        index2score: Dict[int, Dict] = {}
        
        for batch in tqdm(dataloader, desc="Scanning the batches ...", total=len(dataloader)):          
            batch_indexes = batch['indexes']
            batch_prompts = batch['prompts']
            batch_subjects = batch['subjects']
            batch_target_news = batch['target_news']
            batch_ground_truths = batch['ground_truths']
            batch_rephrase_prompts = batch['rephrase_prompts']
            batch_locality_inputs = batch['locality_inputs']
            # batch_portability_inputs = batch['portability_inputs']
            batch_len = batch['batch_len']
            batch_train_ds = None
            
            # from pdb import set_trace; set_trace()
            
            batch_metrics = editor.probe(prompts=batch_prompts, 
                                   target_new=batch_target_news, 
                                   ground_truth=batch_ground_truths,
                                   rephrase_prompts=batch_rephrase_prompts,
                                   locality_inputs=batch_locality_inputs,
                                   verbose=args.verbose,
                                   )
            
            
            all_metrics.extend(batch_metrics)
            
            if save_triple_level_res:
                for index, metric in zip(batch_indexes, batch_metrics):
                    index2score[index] = metric
            
            if cal_NN_score:
                temp_metrics_lists = {
                    '1_1': metrics_1_1_list,
                    '1_N': metrics_1_N_list,
                    'N_1': metrics_N_1_list,
                    'N_N': metrics_N_N_list,
                }
                for index, metrics in zip(batch_indexes, batch_metrics):
                    for data_type, metrics_list in temp_metrics_lists.items():
                        if index in getattr(dataset, f"data_{data_type}_indexes"):
                            metrics_list.append(metrics)
                            break
            
            
        avg_metrics = average_metric_dicts(all_metrics)
        
        if cal_NN_score:
            
            metrics_one_to_one_know = metrics_1_1_list + metrics_N_1_list
            
            metrics_one_to_many_know = metrics_1_N_list + metrics_N_N_list
            
            avg_metrics_one_to_one_know = average_metric_dicts(metrics_one_to_one_know)
            
            avg_metrics_one_to_many_know = average_metric_dicts(metrics_one_to_many_know)
            
            avg_metrics_1_1 = average_metric_dicts(metrics_1_1_list)
            
            avg_metrics_N_1 = average_metric_dicts(metrics_N_1_list)
            
            avg_metrics_1_N = average_metric_dicts(metrics_1_N_list)
            
            avg_metrics_N_N = average_metric_dicts(metrics_N_N_list)
            
            avg_metrics.update({'1_1':avg_metrics_1_1,
                                'N_1':avg_metrics_N_1,
                               '1_N':avg_metrics_1_N,
                               'N_N':avg_metrics_N_N,
                               'one_to_one_know':avg_metrics_one_to_one_know,
                               'one_to_many_know':avg_metrics_one_to_many_know}
                               )
                
        # df = pd.DataFrame(score_dict)
        
        if metrics_save_dir is None or metrics_save_dir == '' or metrics_save_dir == './':
            # metrics_save_dir = f"ret-editing-SnomedCTKnowledgeTriples-{model_name}"
            if use_sampling:
                project_file_name = f"ret-probing-SnomedCTKnowledgeTriples_{model_name}_sampling"
            elif use_eq_sampling:
                project_file_name = f"ret-probing-SnomedCTKnowledgeTriples_{model_name}_eq_sampling"
            elif use_stratified_sampling:
                project_file_name = f"ret-probing-SnomedCTKnowledgeTriples_{model_name}_stratified_sampling"
            else:
                project_file_name = f"ret-probing-SnomedCTKnowledgeTriples_{model_name}"
        else:
            project_file_name = metrics_save_dir
            
        save_dir_path = os.path.join(root_path, 'results', project_file_name)
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        
        use_demonstration = 'demonstration'
        not_use_demonstration = ''
            
        if is_demonstration: 
            json_file_name = f'{frequency_type}_{model_name}_prompt_mode_{prompt_mode}_{use_demonstration}_scores.json'
        else:
            json_file_name = f'{frequency_type}_{model_name}_prompt_mode_{prompt_mode}_scores.json'
        
            
        save_json(avg_metrics, path=os.path.join(save_dir_path, json_file_name), use_indent=True)
        
        if save_triple_level_res:
            if is_demonstration:
                pickle.dump(index2score, open(os.path.join(save_dir_path, f"{frequency_type}_{model_name}_prompt_mode_{prompt_mode}_demonstration_scores.pickle"), mode="wb"))
            else:
                pickle.dump(index2score, open(os.path.join(save_dir_path, f"{frequency_type}_{model_name}_prompt_mode_{prompt_mode}_scores.pickle"), mode="wb"))
    
    
    
    # Test
    # sentence = "COVID-19 is"
    # for triple in tqdm(triples, desc="Scanning all the triples ...", total=len(triples)):
    #     # ============ #
    #     #    BioGPT    #
    #     # ============ #
    #     prompt = prompt_generater.generate(triple)
    #     print("Prompt ...")
    #     print(prompt)
    #     inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
    #     inputs = {k: v.to(device) for k, v in inputs.items()}
    #     with torch.no_grad():
    #         output = model.generate(**inputs, max_length=64)
    #     result = tokenizer.decode(output[0], skip_special_tokens=True)
    #     print(result)
    
    
    
        # ============ #
        #    SapBert   #
        # ============ #

# with torch.no_grad():
#     beam_output = model.generate(**inputs, min_length=100, max_length=1024, num_beams=5, early_stopping=True)
# result = tokenizer.decode(beam_output[0], skip_special_tokens=True)
# print(result)

def sweep(frequency_type_list, 
          model_name, 
          prompt_mode_list, 
          is_demonstration, 
          use_sampling, 
          use_eq_sampling,
          use_stratified_sampling, 
          save_triple_level_res,
          cal_NN_score,
          metrics_save_dir):
    """
        'long_tail': (0,1),
        'medium': (2,10),
        'high': (11, 100),
        'very_high': (101, 1000),
        'popular': (1001, None)
    """
    # frequency_type_list = ['popular', 'very_high', 'high', 'medium', 'long_tail']
    # frequency_type_list = ['popular', 'very_high']
    # model_name_list = ['BioMedLM']
    # model_name_list = ['BioGPT-Large']
    # prompt_mode_list = [0,1,2,3,4]
    # prompt_mode_list = [3]
    
    for frequency_type in frequency_type_list:
        for prompt_mode in prompt_mode_list:
            main(frequency_type=frequency_type, 
                    model_name=model_name, 
                    prompt_mode=prompt_mode, 
                    is_demonstration=is_demonstration, 
                    use_sampling=use_sampling, 
                    use_eq_sampling=use_eq_sampling,
                    use_stratified_sampling=use_stratified_sampling, 
                    save_triple_level_res=save_triple_level_res,
                    cal_NN_score=cal_NN_score,
                    metrics_save_dir=metrics_save_dir)
    


if __name__ == "__main__":

    # triples = triples_selector.f_level_frequency_triples
    # main(triples)
    # main(frequency_type='popular', model_name='BioMedLM', prompt_type_index=0, probing_method='demonstration')
    
    # main(frequency_type='popular', model_name='BioMedLM', prompt_mode=0, is_demonstration=False, use_entity_linking_eval=True)
    
    args = setup_parser()
        
    sweep(frequency_type_list = args.frequency_type,
          model_name = args.model_name, 
          prompt_mode_list=args.prompt_mode,
          is_demonstration=args.use_demonstration,
          use_sampling = args.use_sampling,
          use_eq_sampling = args.use_eq_sampling,
          use_stratified_sampling=args.use_stratified_sampling,
          save_triple_level_res=args.save_triple_level_res,
          cal_NN_score=args.cal_NN_score,
          metrics_save_dir=args.metrics_save_dir
          )
        
    
    


"""
UserWarning: `num_beams` is set to 1. However, `early_stopping` is set to `True` 
-- this flag is only used in beam-based generation modes. You should set `num_beams>1` or unset `early_stopping`.
"""
# with torch.no_grad():
#     output = model.generate(**inputs, min_length=100, max_length=1024)
#     print(output)
# a = tokenizer.decode(output[0], skip_special_tokens=True)
# print(a)

# 'COVID-19 is a global pandemic caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), the causative agent of coronavirus disease 2019 (COVID-19), which has spread to more than 200 countries and territories, including the United States (US), Canada, Australia, New Zealand, the United Kingdom (UK), and the United States of America (USA), as of March 11, 2020, with more than 800,000 confirmed cases and more than 800,000 deaths.'