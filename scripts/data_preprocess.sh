cd "$(dirname "$0")"/../

/nfs/anaconda3/envs/llama/bin/python -m entity_linking_preprocess.umls_entity_linking.py

/nfs/anaconda3/envs/llama/bin/python -m entity_linking_preprocess.generate_CliKT