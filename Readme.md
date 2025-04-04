# Can We Edit LLMs for Long-Tail Biomedical Knowledge?

This repository contains the official implementation of our experiments on biomedical long-tail knowledge. For details, please refer to our paper.

## Introduction
Knowledge editing has emerged as an effective approach for updating large language models (LLMs) by modifying their internal knowledge. 
However, their application to the biomedical domain faces unique challenges due to the long-tailed distribution of biomedical knowledge, where rare and infrequent information is prevalent. In this paper, we conduct the first comprehensive study to investigate the effectiveness of knowledge editing methods for editing long-tail biomedical knowledge. Our results indicate that, while existing editing methods can enhance LLMs' performance on long-tail biomedical knowledge, their performance on long-tail knowledge remains inferior to that on high-frequency popular knowledge, even after editing.
Our further analysis reveals that long-tail biomedical knowledge contains a significant amount of one-to-many knowledge, where one subject and relation link to multiple objects. This high prevalence of one-to-many knowledge limits the effectiveness of knowledge editing in improving LLMs' understanding of long-tail biomedical knowledge, highlighting the need for tailored strategies to bridge this performance gap.

Our work operates in three main stages:
1. **Identifying Long-tail Biomedical Knowledge**: Extract long-tail biomedical knowledge triples from PubMed.
2. **Knowledge Probing on Long-tail Biomedical Knowledge**: Assess LLMs' ability to recall and reason over long-tail biomedical knowledge using tailored evaluation prompts.
3. **Knowledge Editing on Long-tail Biomedical Knowledge**: Apply editing techniques to enhance LLMs' performance on long-tail biomedical knowledge.

edit_bio_long_tail/long-tail_example.pdf
### Overview
![Overview](/long_tail_overview.png)

## Prepare Data
The data used in our experiments are publicly available:
- [PubTator](https://hotpotqa.github.io/)
- [SNOMED CT](https://github.com/Alab-NII/2WikiMultiHopQA)

To preprocess the data and identify long-tail biomedical knowledge:
```bash
python -m entity_linking_preprocess.pubmed_snomedCT_entity_linking
python -m entity_linking_preprocess.generate_CliKT
```

## Experiments on Long-tail Biomedical Knowledge
Our work includes two main experiments:
1. **Knowledge Probing**
2. **Knowledge Editing**


### 1. Knowledge Probing
Run the following command for knowledge probing:
```bash
python -m experiments.knowledge_probing \
    --frequency_type popular high medium long_tail \
    --model_name BioMedLM \
    --prompt_mode 0 \
    --save_triple_level_res \
    --use_demonstration \
    --cal_NN_score \
    --verbose \
    --metrics_save_dir ret-probing-SnomedCTKnowledgeTriples_BioMedLM
```
- **Parameters**:  
  - `--frequency_type`: Specifies the frequency types of knowledge triples to probe. Options include `popular`, `high`, `medium`, and `long_tail`, allowing evaluation across knowledge with varying occurrence frequencies in the pretraining data.  
  - `--model_name`: Specifies the model used for probing. Example: `BioMedLM`.  
  - `--prompt_mode`: Determines the type or structure of the prompt used during probing. Integer values (e.g., `0`) represent predefined prompt templates.  
  - `--save_triple_level_res`: Saves probing results at the triple level, ensuring fine-grained output for each knowledge triple.  
  - `--use_demonstration`: Enables few-shot demonstrations during probing to enhance model performance.  
  - `--cal_NN_score`: Calculates one-to-many scores during probing, providing additional evaluation.  
  - `--verbose`: Enables detailed logging output, aiding in debugging and analysis.  
  - `--metrics_save_dir`: Specifies the output directory for saving probing metrics. Example: `ret-probing-SnomedCTKnowledgeTriples_BioMedLM`. 

### 2. Knowledge Editing
Run the following command for knowledge editing:
```bash
python -m experiments.knowledge_editing \
    --frequency_type popular high medium long_tail \
    --model_name BioMedLM \
    --prompt_mode 0 \
    --editing_method ROME \
    --cal_NN_score \
    --save_triple_level_res \
    --hparams_dir ./hparams/ROME/rome_biomedlm \
    --is_rephrase \
    --is_locality \
    --use_demonstration \
    --metrics_save_dir ret-editing-SnomedCTknoledgeTriples-BioMedLM-ROME
```
- **Parameters**:  
  - `--frequency_type`: Specifies the frequency types of knowledge triples to probe. Options include `popular`, `high`, `medium`, and `long_tail`, allowing evaluation across knowledge with varying occurrence frequencies in the pretraining data.  
  - `--model_name`: Specifies the model used for probing. Example: `BioMedLM`.  
  - `--prompt_mode`: Determines the type or structure of the prompt used during probing. Integer values (e.g., `0`) represent predefined prompt templates.  
  - `--editing_method`: Specifies the knowledge editing method. Example: `ROME`.  
  - `--cal_NN_score`: Calculates one-to-many scores during probing, providing additional evaluation.  
  - `--save_triple_level_res`: Saves probing results at the triple level, ensuring fine-grained output for each knowledge triple.  
  - `--hparams_dir`: Specifies the directory containing hyperparameter configurations for the editing method. Example: `./hparams/ROME/rome_biomedlm`.  
  - `--is_rephrase`: Enables rephrasing during knowledge probing or editing.  
  - `--is_locality`: Ensures locality constraints are applied during knowledge editing.  
  - `--use_demonstration`: Enables few-shot demonstrations during probing to enhance model performance.  
  - `--metrics_save_dir`: Specifies the output directory for saving probing metrics. Example: `ret-editing-SnomedCTknoledgeTriples-BioMedLM-ROME`.  



