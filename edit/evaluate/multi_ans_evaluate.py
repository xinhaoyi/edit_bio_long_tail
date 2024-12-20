from typing import List, Optional, Union

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from ..util.position_ids import get_position_ids
from ..util.padding_side import detect_padding_direction
from .evaluate_utils import test_prediction_acc

def compute_multi_ans_knowledge_avg_acc(
    model,
    hparams: HyperParams,
    tok: AutoTokenizer,
    multi_ans_knowledge_prompts: List[str],
    multi_ans_knowledge_targets: List[str],
    device,
):
    acc_res_for_multi_ans_knowledge = test_prediction_acc(model=model, 
                                  tok=tok, 
                                  hparams=hparams,
                                  prompts=multi_ans_knowledge_prompts,
                                  targets=multi_ans_knowledge_targets,
                                  device=device)
    
    avg_acc = np.mean(acc_res_for_multi_ans_knowledge)
    
    return avg_acc
    
def compute_multi_ans_knowledge_max_acc(
    model,
    hparams: HyperParams,
    tok: AutoTokenizer,
    multi_ans_knowledge_prompts: List[str],
    multi_ans_knowledge_targets: List[str],
    device,
):
    acc_res_for_co_tailed_prompts = test_prediction_acc(model=model, 
                                  tok=tok, 
                                  hparams=hparams,
                                  prompts=multi_ans_knowledge_prompts,
                                  targets=multi_ans_knowledge_targets,
                                  device=device)
    
    max_acc = np.max(acc_res_for_co_tailed_prompts)
    
    return max_acc

def PPL(
    model,
    tok,
    prompt: Union[str, List[str]],
    target_new: Union[str, List[str]],
    device,
):
    if isinstance(prompt, str):
        prompt,target_new = [prompt,], [target_new,]
    full_prompt = [f"{p} {l} <|endoftext|>" for p, l in zip(prompt, target_new)]
    prompt_ids = tok(list(prompt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
    tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()
    for i in range(len(prompt)):
        tokens["labels"][i][:num_prompt_toks[i]] = -100
    tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = -100 # What is this doing?
    batch = {f"{k1}" : v1 for k1, v1 in tokens.items()}
    input_ids = batch["input_ids"][:, :1024]#.to(device)
    if "labels" not in batch:
        target_ids = batch["input_ids"][:, :1024].clone()
    else:
        target_ids = batch["labels"][:, :1024].clone()
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device), labels=target_ids.to(device))
        nll = outputs.loss
    ppl = torch.exp(nll)#.clip(0, 100)
    return ppl.cpu().numpy().tolist()


def compute_perplexity(
    model,
    tok: AutoTokenizer,
    hparams: HyperParams,
    prompt: str,
    target: str,
    device
):
    prompt_target = prompt + ' ' + target
    
    # max_prompt_target_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    # max_prompt_len = max([len(tok.encode(_)) for _ in prompts]) + 1
    
    
    max_prompt_target_len = len(tok.encode(prompt_target))
    max_prompt_len = len(tok.encode(prompt))
    
    if hparams is None or not hasattr(hparams):
        hparams_max_length = 0
    else:
        hparams_max_length = hparams.max_length
    
    max_prompt_target_len = max(max_prompt_target_len, hparams_max_length)
    max_prompt_len = max(max_prompt_len, hparams_max_length)
        
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=max_prompt_target_len,
        return_tensors="pt",
    ).to(f"cuda:{device}")
    
    prompt_tok = tok(
        prompt,
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
        return_tensors="pt",
    )
    
    # BioGPT-Large will modify the position_ids automatically
    if hparams.batch_size > 1 and ('BioMedLM' in hparams.model_name or 'GPT-Neo' in hparams.model_name):
        prompt_target_tok["position_ids"] = get_position_ids(prompt_target_tok["attention_mask"])
        
    
    
    padding_side = detect_padding_direction(tok)
    
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    
    
    if 'left' == padding_side or 'unknown' == padding_side:
        prompt_len = num_pad_toks + num_prompt_toks
    elif 'right':
        prompt_len = num_prompt_toks
        
    
    if padding_side == 'right':
        start_idx = prompt_len
                                                                                                                       
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
    
    # Calculate probabilities from logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
        
    # Adjust indices for extracting target probabilities
    if padding_side == 'right' or padding_side == 'unknown':
        start_idx = num_prompt_toks  # Start from the end of the prompt
        end_idx = start_idx + len(tok.encode(target))  # Exclude padding
    else:  # 'left' padding
        end_idx = prompt_target_tok.input_ids.size(1) - num_pad_toks  # Stop before padding starts
        start_idx = end_idx - len(tok.encode(target))
        
    # Extract probabilities for the target text
    target_probs = probs[0, start_idx:end_idx, :]

    # Corresponding target labels for loss calculation
    target_labels = prompt_target_tok['input_ids'][0, start_idx+1:end_idx+1]

    # Gather the probabilities of the actual next words
    target_log_probs = torch.log(torch.gather(target_probs, 2, target_labels.unsqueeze(2)).squeeze(2))

    # Calculate mean of log probabilities
    mean_log_prob = target_log_probs.mean()

    # Perplexity is the exponential of the negative average log probability
    perplexity = torch.exp(-mean_log_prob).item()

    return perplexity
    