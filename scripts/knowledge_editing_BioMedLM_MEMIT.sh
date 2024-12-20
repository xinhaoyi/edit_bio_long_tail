cd "$(dirname "$0")"/../

/nfs/anaconda3/envs/llmedit/bin/python -m experiments.knowledge_editing \
    --frequency_type popular high medium long_tail \
    --model_name BioMedLM \
    --prompt_mode 0 \
    --editing_method MEMIT \
    --cal_NN_score \
    --save_triple_level_res \
    --hparams_dir ./hparams/MEMIT/memit_biomedlm \
    --is_rephrase \
    --is_locality \
    --use_demonstration \
    --use_stratified_sampling \
    --metrics_save_dir ret-editing-SnomedCTknoledgeTriples-BioMedLM-MEMIT_stratified_sampling \


/nfs/anaconda3/envs/llmedit/bin/python -m experiments.knowledge_editing \
    --frequency_type popular high medium long_tail \
    --model_name BioMedLM \
    --prompt_mode 0 \
    --editing_method MEMIT \
    --cal_NN_score \
    --save_triple_level_res \
    --hparams_dir ./hparams/MEMIT/memit_biomedlm \
    --is_rephrase \
    --is_locality \
    --use_demonstration \
    --use_sampling \
    --metrics_save_dir ret-editing-SnomedCTknoledgeTriples-BioMedLM-MEMIT_natural_sampling \


