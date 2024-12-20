cd "$(dirname "$0")"/../

/nfs/anaconda3/envs/llmedit/bin/python /nfs/long_tail/knowledge_editing/knowledge_editing.py \
    --frequency_type long_tail \
    --model_name BioMedLM \
    --prompt_mode 0 \
    --editing_method MEMIT \
    --use_sampling \
    --save_triple_level_res \
    --hparams_dir ./hparams/MEMIT/biomedlm \
    --metrics_save_dir ./ret-editing-SnomedCTknoledgeTriples-BioMedLM-MEMIT-smapling \

# --is_rephrase \


