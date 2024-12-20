cd "$(dirname "$0")"/../


# BiomedLM
/nfs/anaconda3/envs/llmedit/bin/python -m experiments.knowledge_probing \
    --frequency_type popular high medium long_tail \
    --model_name BioMedLM \
    --prompt_mode 0 \
    --save_triple_level_res \
    --use_demonstration \
    --cal_NN_score \
    --verbose \
    --metrics_save_dir ret-probing-SnomedCTKnowledgeTriples_BioMedLM


# BioGPT
/nfs/anaconda3/envs/llmedit/bin/python -m experiments.knowledge_probing \
    --frequency_type popular high medium long_tail \
    --model_name BioGPT-Large \
    --prompt_mode 0 \
    --save_triple_level_res \
    --cal_NN_score \
    --use_demonstration \
    --verbose \
    --metrics_save_dir ret-probing-SnomedCTKnowledgeTriples_BioGPT-Large


# Llama2
/nfs/anaconda3/envs/llama3/bin/python -m experiments.knowledge_probing \
    --frequency_type popular high medium long_tail \
    --model_name Llama2 \
    --prompt_mode 0 \
    --save_triple_level_res \
    --use_demonstration \
    --cal_NN_score \
    --verbose \
    --metrics_save_dir ret-probing-SnomedCTKnowledgeTriples_Llama2


# GPT-J
/nfs/anaconda3/envs/llmedit/bin/python -m experiments.knowledge_probing \
    --frequency_type popular high medium long_tail \
    --model_name GPT-J \
    --prompt_mode 0 \
    --save_triple_level_res \
    --cal_NN_score \
    --use_demonstration \
    --verbose \
    --metrics_save_dir ret-probing-SnomedCTKnowledgeTriples_GPT-J

