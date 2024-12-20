cd "$(dirname "$0")"/../

/nfs/anaconda3/envs/llmedit/bin/python /nfs/long_tail/knowledge_editing/knowledge_editing.py \
    --frequency_type popular \
    --model_name BioMedLM \
    --prompt_mode 5 \
    --editing_method SERAC \
    --hparams_dir ./hparams/SERAC/biomedlm


