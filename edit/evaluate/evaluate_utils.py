import torch
import numpy as np
import scipy
import nltk
import typing

from ..trainer.algs import SERAC
from ..util.generate import generate_fast
from ..util.padding_side import detect_padding_direction
from ..util.position_ids import get_position_ids
import torch.nn.functional as F
from ..trainer import *


def test_batch_prediction_acc(model, tok, hparams, prompts, target, device, locality=False):
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        if tok.padding_side == 'left':
            ans = torch.argmax(logits, dim=-1)[:, -1].squeeze()
        else:
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)

        ans = ans.squeeze().detach().cpu().numpy().tolist()

        if locality:
            return ans

        return np.mean(np.equal(ans, target))
    
    
    

def test_seq2seq_batch_prediction_acc(model, tok, hparams, prompts, targets, device, locality=False):
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    trg_tok = tok(
        targets,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok['decoder_input_ids'] = trg_tok['input_ids']
    prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            answers = ans.squeeze().detach().cpu().numpy().tolist()
            return answers if type(answers[0]) is list else [answers,]
        return torch.mean((trg_tok['input_ids'][:,:-1] == ans[:,:-1]).float(), dim=-1).detach().cpu().numpy().tolist()


def test_batch_probability_lg_prediction_acc(
    model,
    tok,
    hparams,
    prompts,
    targets,
    device
):
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    try:
        prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts, targets)]
    except Exception as e:
        print(f"Some errors or unmatches in \n \"prompts\"\n{prompts}  \"targets\"\n{targets}")
    
    # tok.encode(_) 拿到的是一系列tok id 最小的是0，所以肯定要+1，来代表长度啊 
    
    if hparams is None or not hasattr(hparams):
        hparams_max_length = 0
        max_prompt_target_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
        max_prompt_len = max([len(tok.encode(_)) for _ in prompts]) + 1
        max_target_len = max([len(tok.encode(f" {target}")) for target in targets]) + 1
    else:
        hparams_max_length = hparams.max_length
            
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=max(hparams_max_length, max_prompt_target_len),
        return_tensors="pt",
    ).to(f"cuda:{device}")
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(hparams_max_length, max_prompt_len),
        return_tensors="pt",
    )
    target_tok = tok(
        f" {targets}",
        padding=True,
        truncation=True,
        max_length=max(hparams_max_length, max_target_len),
        return_tensors="pt",
    )
    
    # BioGPT-Large will modify the position_ids automatically
    # position_ids
    if hasattr(hparams, 'batch_size') and hparams.batch_size > 1 and ('BioMedLM' in hparams.model_name or 'GPT-Neo' in hparams.model_name):
        prompt_target_tok["position_ids"] = get_position_ids(prompt_target_tok["attention_mask"])
    
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    
    padding_side = detect_padding_direction(tok)
    
    if 'left' == padding_side or 'unknown' == padding_side:
        prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    elif 'right':
        prompt_len = num_prompt_toks
    
    # 我希望拿到每一个bacth中，对应的target的所有tok id
    # list 代表了batch size， 内部的tensor装的是当前target所有真实的token id
    # actual_tok_ids_list: typing.List[torch.Tensor] = [target_tok['input_ids'][i, target_tok['attention_mask'][i].nonzero().squeeze(1)] for i in range(target_tok['input_ids'].size(0))]
    
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
    
    probabilities_for_prompts = np.zeros((logits.size(0),), dtype=np.float32)
    
    for prompt_idx in range(logits.size(0)):
        # 当前真实答案的token_id
        target_actual_tok_ids = target_tok['input_ids'][prompt_idx][target_tok['attention_mask'][prompt_idx].nonzero().squeeze(1)]
        probability_gen = 0.0
        for target_tok_idx in range(len(target_actual_tok_ids)):
            cur_token = target_actual_tok_ids[target_tok_idx]
            # 我的答案的第一个token，应该是拼上去的“ ”，它对应的logits
            # 是做的next token prediction，也就是这个logits要预测出label的第一个token
            probability_tok = torch.nn.functional.log_softmax(logits[prompt_idx, prompt_len + target_tok_idx, : ], dim= 0)[cur_token].detach().cpu().item()
            probability_gen += probability_tok
        probability_gen = probability_gen / len(target_actual_tok_ids)
        probabilities_for_prompts[prompt_idx] = probability_gen
    
    return probabilities_for_prompts
    
        

def test_probability_lg_prediction_acc(
    model,
    tok,
    hparams,
    prompt: str,
    target: str,
    device
):
    assert not isinstance(prompt, str) or not isinstance(target, str) or print(f"The \"prompt\" {prompt} and \"target\" should be \"str\"") 
    
    prompt_target = prompt + ' ' + target
            
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(f"cuda:{device}")
    prompt_tok = tok(
        prompt,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    target_tok = tok(
        f" {target}",
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    
    # BioGPT-Large will modify the position_ids automatically
    # position_ids
    if hasattr(hparams, 'batch_size') and hparams.batch_size > 1 and ('BioMedLM' in hparams.model_name or 'GPT-Neo' in hparams.model_name):
        prompt_target_tok["position_ids"] = get_position_ids(prompt_target_tok["attention_mask"])
    
    num_prompt_toks = int((prompt_tok['input_ids'] != tok.pad_token_id).sum())
    num_pad_toks = int((prompt_target_tok['input_ids'].cpu() == tok.pad_token_id).sum())
    
    padding_side = detect_padding_direction(tok)
    
    if 'left' == padding_side or 'unknown' == padding_side:
        prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    elif 'right':
        prompt_len = num_prompt_toks
    
    # 我希望拿到每一个bacth中，对应的target的所有tok id
    # list 代表了batch size， 内部的tensor装的是当前target所有真实的token id
    # actual_tok_ids_list: typing.List[torch.Tensor] = [target_tok['input_ids'][i, target_tok['attention_mask'][i].nonzero().squeeze(1)] for i in range(target_tok['input_ids'].size(0))]
    
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
    
    probabilities_for_prompts = np.zeros((logits.size(0),), dtype=np.float32)
    
    for prompt_idx in range(logits.size(0)):
        # 当前真实答案的token_id
        target_actual_tok_ids = target_tok['input_ids'][prompt_idx][target_tok['attention_mask'][prompt_idx].nonzero().squeeze(1)]
        probability_gen = 0.0
        for target_tok_idx in range(len(target_actual_tok_ids)):
            cur_token = target_actual_tok_ids[target_tok_idx]
            # 我的答案的第一个token，应该是拼上去的“ ”，它对应的logits
            # 是做的next token prediction，也就是这个logits要预测出label的第一个token
            probability_tok = torch.nn.functional.log_softmax(logits[prompt_idx, prompt_len + target_tok_idx, : ], dim= 0)[cur_token].item()
            probability_gen += probability_tok
        probability_gen = probability_gen / len(target_actual_tok_ids)
        probabilities_for_prompts[prompt_idx] = probability_gen
    
    return probabilities_for_prompts


def test_prediction_acc(model, tok, hparams, prompts, targets, device, locality=False):
    if hparams is None:
        hparams.max_length = 250
    
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    try:
        prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    except Exception as e:
        from pdb import set_trace; set_trace()
        
    if hparams is None or not hasattr(hparams, 'max_length'):
        max_prompt_target_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
        max_prompt_len = max([len(tok.encode(_)) for _ in prompts]) + 1
        # max_target_len = max([len(tok.encode(f"{target} ")) for target in targets]) + 1
    else:
        max_prompt_target_len = max_prompt_len = hparams.max_length
            
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=max_prompt_target_len,
        return_tensors="pt",
    ).to(f"cuda:{device}")
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
        return_tensors="pt",
    )
    
    # from pdb import set_trace; set_trace()
    
    # batch_size, seq_len = attention_mask.shape
    # BioGPT-Large will modify the position_ids automatically
    # and ('BioMedLM' in hparams.model_name or 'GPT-Neo' in hparams.model_name)
    if hasattr(hparams, 'batch_size') and hparams.batch_size > 1 and (prompt_target_tok["attention_mask"].shape[0] > 1):
        temp_position_ids = get_position_ids(prompt_target_tok["attention_mask"])
        if temp_position_ids is not None:
            prompt_target_tok["position_ids"] = temp_position_ids
            prompt_target_tok.to(f"cuda:{device}")
    
    # from pdb import set_trace; set_trace()
    if "llama2" in hparams.model_name.lower() or "biogpt" in hparams.model_name.lower():
        if 'token_type_ids' in prompt_target_tok:
            prompt_target_tok.pop('token_type_ids')
        if 'position_ids' in prompt_target_tok:
            prompt_target_tok.pop('position_ids')
        
    
    # from pdb import set_trace; set_trace()
        
    
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers,prompt_len,left=True)
        labels = slice_list(labels,prompt_len,left=False)
        # if len(answers) > 2:
        #     from pdb import set_trace; set_trace()
        if locality:
            return answers if type(answers[0]) is list else [answers,]
        
        # from pdb import set_trace; set_trace()
        if isinstance(answers[0], list):
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res
        else:
            return [np.mean(np.equal(answers, labels))]



def test_generation_quality_serac(
    model,
    tok,
    prefixes: typing.List[str],
    max_out_len: int,       
):
    #only single case
    prompt_tok = tok(
        prefixes,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    prompt_tok_length=len(prompt_tok['input_ids'])
    gen_texts=model.generate(**prompt_tok,max_new_tokens=256)
    if isinstance(model,SERAC):
        gen_texts=tok.decode(gen_texts[prompt_tok_length:])
        gen_texts=[gen_texts]
        print(len(gen_texts))
    else:
        gen_texts=tok.decode(gen_texts[prompt_tok_length:])
        gen_texts=[gen_texts]
        print(len(gen_texts))      
    ngram_entropy = n_gram_entropy(gen_texts, return_list=True)


    ret = {
        "ngram_entropy": ngram_entropy
    }
    return ret

def test_generation_quality(
    model,
    tok,
    prefixes: typing.List[str],
    max_out_len: int,
    # consistency_texts: typing.List[str],
    # essence_texts: typing.List[str],
    # vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=max_out_len,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    # consistency_tfidf = tfidf_similarity(
    #     " ".join(gen_texts), " ".join(consistency_texts), vec
    # )

    ret = {
        "ngram_entropy": ngram_entropy,
        # "reference_score": consistency_tfidf,
        # "text": gen_texts,
    }

    # if len(essence_texts) > 0:
    #     ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
    #     ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def PPL(
    model,
    tok,
    prompt: typing.Union[str, typing.List[str]],
    target_new: typing.Union[str, typing.List[str]],
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


def verify_answer(model_answer, correct_answer):
    if type(correct_answer) is str:
        correct_answer = [[correct_answer]]
    for answer in correct_answer:
        if True not in [possible_answer in model_answer for possible_answer in answer]:
            return False
    return True


def answer_match(
    model,
    tok,
    prompt: str,
    target_new: str,
    device,
):
    inputs = tok.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs, temperature=0, max_new_tokens=30)
    predict = tok.decode(outputs[0], skip_special_tokens=True)

    return verify_answer(predict,target_new)


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

def gather_log_probs(logits, labels):
    # print(f"labels.shape: {labels.shape} , logits.shape[:-1] :{logits.shape[:-1]}")
    assert labels.dim() == logits.dim() - 1
    assert labels.shape == logits.shape[:-1]
    return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()


def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels


def es_sent(pre_logits, edit_logits, q_mask, labels, same_mask):
    
    _, targ = mask_hf_labels(labels)

    pos_mask = same_mask.unsqueeze(-1) * q_mask 
    neg_mask = (~same_mask).unsqueeze(-1) * q_mask 
        
    pre_token_log_probs = gather_log_probs(pre_logits, targ)
    edit_token_log_probs = gather_log_probs(edit_logits, targ)

    mean_pos_pre = masked_mean(pre_token_log_probs, pos_mask)
    mean_pos_edit = masked_mean(edit_token_log_probs, pos_mask)
    mean_neg_edit = masked_mean(edit_token_log_probs, neg_mask)

    z_sent = (mean_pos_edit - mean_neg_edit).sigmoid()
    z_topic_raw = (mean_pos_edit - mean_pos_pre).exp()
    z_topic = min(1, z_topic_raw)

    es_sent = z_sent * z_topic
    return es_sent
        

def kl_loc_loss(pre, post, mask=None):
    
    pre = pre.to(torch.float32).contiguous()
    post = post[:,-pre.shape[1]:,:].to(torch.float32).contiguous()
    
    sequence = pre.dim() == 3
    pre_ = pre.view(-1, pre.shape[-1])
    post_ = post.view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        # print("sequence")
        if pre_.shape[-1] > 1:
            assert mask is not None
            mask_ = mask.view(pre_.shape[0])
            kl = (pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))).sum(-1)
            return (kl * mask_).sum() / mask_.sum()

    raise NotImplementedError
