cd "$(dirname "$0")"/../

# CUDA_VISIBLE_DEVICES=1 /nfs/anaconda3/envs/llmedit/bin/python -m experiments.knowledge_editing \
#     --frequency_type popular high medium long_tail \
#     --model_name BioMedLM \
#     --prompt_mode 0 \
#     --editing_method FT \
#     --cal_NN_score \
#     --save_triple_level_res \
#     --hparams_dir ./hparams/FT/biomedlm-ft-14 \
#     --is_rephrase \
#     --is_locality \
#     --use_demonstration \
#     --use_stratified_sampling \
#     --metrics_save_dir ret-editing-SnomedCTknoledgeTriples-BioMedLM-FT-14_stratified_smapling \


CUDA_VISIBLE_DEVICES=1 /nfs/anaconda3/envs/llmedit/bin/python -m experiments.knowledge_editing \
    --frequency_type popular high medium long_tail \
    --model_name BioMedLM \
    --prompt_mode 0 \
    --editing_method FT \
    --cal_NN_score \
    --save_triple_level_res \
    --hparams_dir ./hparams/FT/biomedlm-ft-14 \
    --is_rephrase \
    --is_locality \
    --use_demonstration \
    --use_sampling \
    --metrics_save_dir ret-editing-SnomedCTknoledgeTriples-BioMedLM-FT-14_natural_smapling \