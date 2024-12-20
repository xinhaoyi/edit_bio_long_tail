import json
import pickle
import sys

import numpy as np

sys.path.append("/nfs")
sys.path.append("/nfs/long_tail")
sys.path.append("/nfs/general")

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from tqdm import tqdm

from itertools import zip_longest

import argparse
from argparse import Namespace
from typing import Any, Dict, List
from dsets.knowledge_triples_dataset import SnomedCTKnowledgeTriplesDataset
from utils.collators import SnomedCTKnowledgeTriplesEditingCollator

from edit.editors.my_editor import MyEditor

import torch

from entity_linking_preprocess.snomedCT_entity_linking import sapbert_entity_linking
from dataclasses import dataclass
from edit.models.ike.util import encode_ike_facts
from sentence_transformers import SentenceTransformer

from utils.knowledge_utils import average_metric_dicts 

from utils.my_utils import get_dataloader, save_json

from edit.models.ft import FTHyperParams
from edit.models.ike import IKEHyperParams
from edit.models.rome import ROMEHyperParams
from edit.models.memit import MEMITHyperParams
from edit.models.serac import SERACHparams
from edit.models.mend import MENDHyperParams

from config import PathConfig


root_path = str(PathConfig.BASE_DIR)

def setup_parser():
    # nargs='+' make it available to receive one or more values (a list)
    parser = argparse.ArgumentParser(description="Run sweep function with command line arguments")
    parser.add_argument("--frequency_type", nargs='+', required=False, default= ['popular', 'very_high', 'high', 'medium', 'long_tail'], help="List of frequency types")
    parser.add_argument("--model_name", type=str, required=False, default='BioGPT-Large')
    parser.add_argument("--prompt_mode", nargs='+', type=int, required=False, default=[0,1,2,3,4])
    parser.add_argument("--use_demonstration", action="store_true")
    parser.add_argument("--use_sampling", action="store_true")
    parser.add_argument("--use_eq_sampling", action="store_true")
    parser.add_argument("--use_stratified_sampling", action="store_true")
    parser.add_argument("--editing_method", required=True, type=str)
    
    parser.add_argument("--is_rephrase", action="store_true")
    parser.add_argument("--is_locality", action="store_true")
    parser.add_argument("--is_portability", action="store_true")
    
    parser.add_argument("--is_sequential_editing", action="store_true")
    
    parser.add_argument("--use_entity_linking_eval", action="store_true")
    parser.add_argument("--cal_NN_score", action="store_true")
    parser.add_argument("--save_triple_level_res", action="store_true")
    
    parser.add_argument("--edit_batch_size", type=int, required=False, default=8)
    
    parser.add_argument("--verbose", action="store_true")
    
    parser.add_argument("--hparams_dir", nargs='+', type=str, required=False, default=['./hparams/ROME/gpt2-xl'])
    parser.add_argument("--metrics_save_dir", default='./', type=str)
    
    return parser.parse_args()

# args: Namespace

def get_hyperparams(editing_method: str, hparams_dir: str):
    """
        hparams_dir = './hparams/ROME'
    """
    if editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif editing_method == 'SERAC':
        editing_hparams = SERACHparams
    elif editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    else:
        raise NotImplementedError
    
    # hparams_dir = ./hparams/ROME/gpt2-xl
    return editing_hparams.from_hparams(hparams_dir)


def merge_pickle_files(frequency_type, model_name, prompt_mode, demonstration_str, file_counter, save_dir_path):
    # Generate the file pattern to match the files to be merged
    file_pattern = f"{frequency_type}_{model_name}_prompt_mode_{prompt_mode}{demonstration_str}_index2score_"
    merged_data = {}

    # Iterate through each file from 0 to file_counter
    for i in range(file_counter + 1):
        file_name = f"{file_pattern}{i}.pickle"
        
        file_path = os.path.join(save_dir_path, file_name)
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Load data from each pickle file and merge into the dictionary
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                merged_data.update(data)  # Assuming data is a dictionary-like structure
    
    # from pdb import set_trace; set_trace()
    # Save the merged data to a new file
    merged_file_name = f"{frequency_type}_{model_name}_prompt_mode_{prompt_mode}{demonstration_str}_index2score.pickle"
    pickle.dump(merged_data, open(os.path.join(save_dir_path, merged_file_name), mode="wb"))



def main(args: Namespace, 
         frequency_type: str, 
         model_name: str, 
         editing_method: str, 
         prompt_mode: int, 
         hparams_dir: str,
         use_sampling: bool = True,
         use_eq_sampling: bool = False,
         use_stratified_sampling: bool = False,
         cal_NN_score: bool = True,
         is_example=False,
         save_triple_level_res: bool = False):
    print("\n")
    print("\n")
    print(f"####### {frequency_type} #######")
    print(f"####### Model: {model_name} #######")
    print(f"####### Editing Method: {editing_method} #######")
    print(f"####### Prompt Mode = {prompt_mode} #######")
    if args.use_demonstration:
        print(f"####### Use Demonstration #######")
    else:
        print(f"####### Don't Use Demonstration #######")
    
    if args.use_entity_linking_eval:
        print("####### Contain Entity Linking Evaluation #######")
    else:
        print("####### Don't Contain Entity Linking Evaluation #######")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    metrics_save_dir = args.metrics_save_dir
    if metrics_save_dir is None or metrics_save_dir == '' or metrics_save_dir == './':
        # metrics_save_dir = f"ret-editing-SnomedCTKnowledgeTriples-{model_name}"
        if use_sampling:
            project_file_name = f"ret-editing-SnomedCTKnowledgeTriples_{editing_method}_{model_name}_dc_sampling"
        elif use_eq_sampling:
            project_file_name = f"ret-editing-SnomedCTKnowledgeTriples_{editing_method}_{model_name}_eq_sampling"
        elif use_stratified_sampling:
            project_file_name = f"ret-editing-SnomedCTKnowledgeTriples_{editing_method}_{model_name}_stratified_sampling"
        else:
            project_file_name = f"ret-editing-SnomedCTKnowledgeTriples_{editing_method}_{model_name}"
    else:
        project_file_name = metrics_save_dir
    
    save_dir_path = os.path.join(root_path, 'results', project_file_name)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    
    # load dataset & collator
    if use_sampling == use_eq_sampling == True:
        use_eq_sampling = False
        
    frequency_dict = {
                'zero': (0,1),
                'long_tail': (1,10),
                'medium': (10,100),
                'high': (100, 1000),
                'popular': (1000, None)
            }
    
    
    if use_sampling:
        dataset = SnomedCTKnowledgeTriplesDataset(frequency_type=frequency_type, frequency_dict=frequency_dict, save_data_file_name='clinic_knowledge_triples_dc_sampled.json', is_sample=True)
    elif use_eq_sampling:
        dataset = SnomedCTKnowledgeTriplesDataset(frequency_type=frequency_type, frequency_dict=frequency_dict, save_data_file_name='clinic_knowledge_triples_eq-sampled.json', is_sample=True)
    elif use_stratified_sampling:
        dataset = SnomedCTKnowledgeTriplesDataset(frequency_type=frequency_type, frequency_dict=frequency_dict, save_data_file_name='clinic_knowledge_triples_stratified_sampled.json', is_sample=True)
    else:
        dataset = SnomedCTKnowledgeTriplesDataset(frequency_type=frequency_type, frequency_dict=frequency_dict, save_data_file_name='clinic_knowledge_triples.json', is_sample=True, random_select_datasize=51231)
    
    collator = SnomedCTKnowledgeTriplesEditingCollator(prompt_mode=prompt_mode,
                                                       is_demonstration=args.use_demonstration,
                                                       is_rephrase=args.is_rephrase,
                                                       is_locality=args.is_locality,
                                                       add_demonstration_to_origin_prompt=False,
                                                       )
    
    dataloader = get_dataloader(-1, dataset, batch_size=8, shuffle=False, collate_fn=collator)
    
    hparams = get_hyperparams(editing_method=editing_method, hparams_dir=hparams_dir)
    
    if args.editing_method == 'IKE':
        # train_data_path = os.path.join(args.data_dir, 'zsre_mend_train_10000.json')
        # train_ds = ZsreDataset(train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None
    
    editor = MyEditor.from_hparams(hparams)
    if args.metrics_save_dir is None or args.metrics_save_dir == '' or args.metrics_save_dir == './':
        metrics_save_dir = f"ret-editing-SnomedCTKnowledgeTriples-{model_name}"
        args.metrics_save_dir = metrics_save_dir
        
    # if not os.path.exists(args.metrics_save_dir):
    #     os.makedirs(args.metrics_save_dir)
        
    
    if args.use_demonstration: 
        demonstration_str = "_demonstration"
    else:
        demonstration_str = ''
    
    all_metrics = []
    index2score = {}
    
    file_counter = 0  # File Counter Initialisation
    has_saved_batches = False  # Whether or not the mark is kept in batches
    
    if cal_NN_score:
        metrics_1_1_list: List[Dict] = []
        metrics_1_N_list: List[Dict] = []
        metrics_N_1_list: List[Dict] = []
        metrics_N_N_list: List[Dict] = []
    
    for batch in tqdm(dataloader, desc="Scanning the batches ...", total=len(dataloader)):
        batch_indexes = batch['indexes']
        batch_prompts = batch['prompts']
        batch_subjects = batch['subjects']
        batch_target_news = batch['target_news']
        batch_ground_truths = batch['ground_truths']
        batch_rephrase_prompts = batch['rephrase_prompts']
        batch_locality_inputs = batch['locality_inputs']
        batch_portability_inputs = batch['portability_inputs']
        batch_len = batch['batch_len']
        batch_train_ds = None
        
        # from pdb import set_trace; set_trace()
        
        batch_demonstrated_prompts = batch['demonstrated_prompts'] if 'demonstrated_prompts' in batch else None
        
        
        # from pdb import set_trace; set_trace()
        
        # ! delete
        # from pdb import set_trace; set_trace()
        
        """
        {
            'prompts': prompts,
            'subjects': subjects,
            'targte_news': target_news,
            'rephrase_prompts': rephrase_prompts,
            'locality_inputs': locality_inputs,
            'portability_inputs': portability_inputs,
            'batch_len': len(batch)
        }
        """
        # if args.is_sequential_editing:
        batch_metrics, edited_model, _ = editor.edit(
            prompts=batch_prompts,
            rephrase_prompts=batch_rephrase_prompts,
            target_new=batch_target_news,
            ground_truth=batch_ground_truths,
            subject=batch_subjects,
            train_ds=batch_train_ds,
            locality_inputs=batch_locality_inputs,
            portability_inputs=batch_portability_inputs,
            keep_original_weight=True,
            demonstrated_prompts=batch_demonstrated_prompts,
        )
        """
        [{
            'pre': {'rewrite_acc': [0.0], 'portability': {}},
            'case_id': 0,
            'requested_rewrite': {'prompt': 'What is the finding site of Female infertility due to ovulatory disorder?', 'target_new': 'Ovary', 'ground_truth': 'Ovary', 'portability': {}, 'locality': {}, 'subject': 'Female infertility due to ovulatory disorder'},
            'time': 9.372345447540283, 
            'post': {'rewrite_acc': [0.5], 'locality': {}, 'portability': {}}
        }]
        
        [{
            'pre': {'rewrite_acc': [0.0], 'portability': {}}, 
            'post': {'rewrite_acc': [0.5], 'locality': {}, 'portability': {}}
        }]
        
        """
        
        # from pdb import set_trace; set_trace()
        batch_metrics = [{'pre': metric['pre'] ,'post': metric['post']} for metric in batch_metrics]
        # metrics = [{'post': metric['post']} for metric in metrics]
        all_metrics.extend(batch_metrics)
        
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
        
        if save_triple_level_res:
            for index, metric in zip(batch_indexes, batch_metrics):
                index2score[index] = metric
            
            # Check that preservation conditions are met
            if len(index2score) > 20000:
                pickle.dump(index2score, open(os.path.join(save_dir_path, f"{frequency_type}_{model_name}_prompt_mode_{prompt_mode}{demonstration_str}_index2score_{file_counter}.pickle"), mode="wb"))
                index2score.clear()  # Empty the dictionary
                file_counter += 1  # Increase the file counter
                has_saved_batches = True  # Marks have been saved in batches
        
    avg_metrics = average_metric_dicts(dict_list=all_metrics)
    
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
    
    # if save_triple_level_res:
    #     if args.use_demonstration:
    #         demonstration_str = "use_demonstration"
    #         pickle.dump(index2score, open(os.path.join(args.metrics_save_dir, f'{frequency_type}_{model_name}_prompt_mode_{prompt_mode}_{demonstration_str}_index2score.pickle'), mode="wb"))
    #     else:
    #         pickle.dump(index2score, open(os.path.join(args.metrics_save_dir, f'{frequency_type}_{model_name}_prompt_mode_{prompt_mode}_index2score.pickle'), mode="wb"))
    
            
    # save_dir_path = os.path.join(root_path, 'results', project_file_name)
    # if not os.path.exists(save_dir_path):
    #     os.makedirs(save_dir_path)
        
    # if args.use_demonstration: 
    #     demonstration_str = "_demonstration"
    # else:
    #     demonstration_str = ''
        
    json_file_name = f'{frequency_type}_{editing_method}_{model_name}_prompt_mode_{prompt_mode}{demonstration_str}_scores.json'
    
    
    save_json(avg_metrics, path=os.path.join(save_dir_path, json_file_name), use_indent=True)
    
    if save_triple_level_res and index2score:
            
        if has_saved_batches:
            # Already saved in batches, use filenames with numbers
            file_name = f"{frequency_type}_{model_name}_prompt_mode_{prompt_mode}{demonstration_str}_index2score_{file_counter}.pickle"
        else:
            # Never saved in batches, using filenames without numbers
            file_name = f"{frequency_type}_{model_name}_prompt_mode_{prompt_mode}{demonstration_str}_index2score.pickle"
        pickle.dump(index2score, open(os.path.join(save_dir_path, file_name), mode="wb"))
        # pickle.dump(index2score, open(os.path.join(save_dir_path, f"{frequency_type}_{model_name}_prompt_mode_{prompt_mode}{demonstration_str}_index2score.pickle"), mode="wb"))
        
    if save_triple_level_res and has_saved_batches:
        merge_pickle_files(frequency_type, model_name, prompt_mode, demonstration_str, file_counter, save_dir_path)

        
    
def sweep(args, 
          frequency_type_list, 
          model_name, 
          prompt_mode_list, 
          hparams_dirs):
    for frequency_type in frequency_type_list:
        for hparams_dir in hparams_dirs:
            for prompt_mode in prompt_mode_list:
                main(args=args, 
                     frequency_type=frequency_type, 
                     model_name=model_name, 
                     editing_method=args.editing_method, 
                     prompt_mode=prompt_mode,
                     use_sampling=args.use_sampling,
                     use_eq_sampling=args.use_eq_sampling,
                     use_stratified_sampling=args.use_stratified_sampling,
                     cal_NN_score=args.cal_NN_score,
                     save_triple_level_res=args.save_triple_level_res,
                     hparams_dir=hparams_dir)

            
if __name__ == "__main__":

    # triples = triples_selector.f_level_frequency_triples
    # main(triples)
    # main(frequency_type='popular', model_name='BioMedLM', prompt_type_index=0, probing_method='demonstration')
    
    # main(frequency_type='popular', model_name='BioMedLM', prompt_mode=0, is_demonstration=False, use_entity_linking_eval=True)
    
    args = setup_parser()
        
    sweep(args=args, 
          frequency_type_list = args.frequency_type,
          model_name = args.model_name, 
          prompt_mode_list=args.prompt_mode,
          hparams_dirs=args.hparams_dir
          )
