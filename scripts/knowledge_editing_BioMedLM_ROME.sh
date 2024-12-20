cd "$(dirname "$0")"/../

# /nfs/anaconda3/envs/llmedit/bin/python -m experiments.knowledge_editing \
#     --frequency_type popular high medium long_tail \
#     --model_name BioMedLM \
#     --prompt_mode 0 \
#     --editing_method ROME \
#     --cal_NN_score \
#     --save_triple_level_res \
#     --hparams_dir ./hparams/ROME/rome_biomedlm \
#     --is_rephrase \
#     --is_locality \
#     --use_demonstration \
#     --use_stratified_sampling \
#     --metrics_save_dir ret-editing-SnomedCTknoledgeTriples-BioMedLM-ROME_stratified_sampling \

CUDA_VISIBLE_DEVICES=1 /nfs/anaconda3/envs/llmedit/bin/python -m experiments.knowledge_editing \
    --frequency_type popular high medium long_tail \
    --model_name BioMedLM \
    --prompt_mode 0 \
    --editing_method ROME \
    --cal_NN_score \
    --save_triple_level_res \
    --hparams_dir ./hparams/ROME/rome_biomedlm \
    --is_rephrase \
    --is_locality \
    --use_demonstration \
    --use_sampling \
    --metrics_save_dir ret-editing-SnomedCTknoledgeTriples-BioMedLM-ROME_natural_sampling \

# --is_rephrase \
# --frequency_type very_high high medium long_tail \
# --use_sampling \

# --use_stratified_sampling \


