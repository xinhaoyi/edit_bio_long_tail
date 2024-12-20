cd "$(dirname "$0")"/../

CUDA_VISIBLE_DEVICES=0 /nfs/anaconda3/envs/llmedit/bin/python -m experiments.knowledge_editing \
    --frequency_type popular high medium long_tail \
    --model_name Llama-2 \
    --prompt_mode 0 \
    --editing_method MEMIT \
    --cal_NN_score \
    --save_triple_level_res \
    --hparams_dir ./hparams/MEMIT/memit_llama2-7b \
    --is_rephrase \
    --is_locality \
    --use_demonstration \
    --use_stratified_sampling \
    --metrics_save_dir ret-editing-SnomedCTknoledgeTriples-Llama-2-7b-MEMIT_stratified_sampling \

CUDA_VISIBLE_DEVICES=0 /nfs/anaconda3/envs/llmedit/bin/python -m experiments.knowledge_editing \
    --frequency_type popular high medium long_tail \
    --model_name Llama-2 \
    --prompt_mode 0 \
    --editing_method MEMIT \
    --cal_NN_score \
    --save_triple_level_res \
    --hparams_dir ./hparams/MEMIT/memit_llama2-7b \
    --is_rephrase \
    --is_locality \
    --use_demonstration \
    --use_sampling \
    --metrics_save_dir ret-editing-SnomedCTknoledgeTriples-Llama-2-7b-MEMIT_natural_sampling \

# --is_rephrase \
# --frequency_type very_high high medium long_tail \
# --use_sampling \

# --use_stratified_sampling \


