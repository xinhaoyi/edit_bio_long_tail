CUDA_VISIBLE_DEVICES=3 /nfs/anaconda3/envs/llama/bin/python /nfs/long_tail/knowledge_editing/knowledge_probing.py \
    --frequency_type popular very_high high medium long_tail \
    --model_name GPT-Neo \
    --prompt_mode 0 1 2 3 4 5
    # --prompt_mode 0 1 2 3 4

CUDA_VISIBLE_DEVICES=3 /nfs/anaconda3/envs/llama/bin/python /nfs/long_tail/knowledge_editing/knowledge_probing.py \
    --frequency_type popular very_high high medium long_tail \
    --model_name GPT-Neo \
    --prompt_mode 0 1 2 3 4 5 \
    --use_demonstration

    # --prompt_mode 0 1 2 3 4 \