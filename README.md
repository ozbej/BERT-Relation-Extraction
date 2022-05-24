# BERT(S) for Relation Extraction

This is a fork project of [BERT-Relation-Extraction](https://github.com/plkmo/BERT-Relation-Extraction) repo modified for the [TermFrame](https://termframe.ff.uni-lj.si/) dataset.

## Installation
1. Clone this repo and change dir into it
2. Run `pip install -r requirements.txt`
3. Run `python -m spacy download en` and `python -m spacy download en_core_web_lg`

## Run the model
`python main_task.py --detect_entities 0 --annotated 1 --train_data "train_path.txt" --test_data "test_path.txt" --infer_data "infer_data.txt"`