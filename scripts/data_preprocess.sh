cd "$(dirname "$0")"/../

/nfs/anaconda3/envs/llama/bin/python -m entity_linking_preprocess.pubmed_snomedCT_entity_linking

/nfs/anaconda3/envs/llama/bin/python -m entity_linking_preprocess.generate_CliKT