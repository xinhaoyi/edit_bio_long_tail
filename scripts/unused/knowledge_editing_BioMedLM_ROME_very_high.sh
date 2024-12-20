cd "$(dirname "$0")"/../

/nfs/anaconda3/envs/llmedit/bin/python /nfs/long_tail/knowledge_editing/knowledge_editing.py \
    --frequency_type very_high \
    --model_name BioMedLM \
    --prompt_mode 0 \
    --editing_method ROME \
    --use_stratified_sampling \
    --cal_NN_score \
    --save_triple_level_res \
    --hparams_dir ./hparams/ROME/biomedlm-rome-14 \
    --is_rephrase \
    --metrics_save_dir ./ret-editing-SnomedCTknoledgeTriples-BioMedLM-ROME-14-sampling-stratified \

# --is_rephrase \
# --frequency_type very_high high medium long_tail \


