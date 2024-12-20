/nfs/anaconda3/envs/llmedit/bin/python /nfs/long_tail/knowledge_editing/knowledge_probing.py \
    --frequency_type popular very_high high medium long_tail \
    --model_name BioGPT-Large \
    --prompt_mode 0 1 2 3 4 5 \
    --save_triple_level_res \
    --use_sampling
    # --prompt_mode 0 1 2 3 4

/nfs/anaconda3/envs/llmedit/bin/python /nfs/long_tail/knowledge_editing/knowledge_probing.py \
    --frequency_type popular very_high high medium long_tail \
    --model_name BioGPT-Large \
    --prompt_mode 0 1 2 3 4 5 \
    --use_sampling \
    --save_triple_level_res \
    --use_demonstration