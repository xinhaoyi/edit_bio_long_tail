cd "$(dirname "$0")"/../

/nfs/anaconda3/envs/llmedit/bin/python /nfs/long_tail/knowledge_editing/knowledge_editing.py \
    --frequency_type popular very_high high medium long_tail \
    --model_name BioMedLM \
    --prompt_mode 0 1 2 3 4 5 \
    --editing_method IKE \
    --use_sampling \
    --hparams_dir ./hparams/IKE/biomedlm \
    --metrics_save_dir ./ret-editing-SnomedCTknoledgeTriples-BioMedLM-IKE-smapling \


