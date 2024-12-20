cd "$(dirname "$0")"/../

/nfs/anaconda3/envs/llmedit/bin/python -m experiments.knowledge_editing \
    --frequency_type popular \
    --model_name BioGPT-Large \
    --prompt_mode 0 \
    --editing_method ROME \
    --cal_NN_score \
    --save_triple_level_res \
    --hparams_dir ./hparams/ROME/rome_biogpt-large \
    --is_rephrase \
    --is_locality \
    --use_demonstration \
    --metrics_save_dir ret-editing-SnomedCTknoledgeTriples-BioGPT-Large-ROME \

# --is_rephrase \
# --frequency_type very_high high medium long_tail \

# --use_stratified_sampling \


