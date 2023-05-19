# Similarity-Based Content Scoring - A more Classroom-Suitable Alternative to Instance-Based Scoring?

This is the repository to our 2023 ACL Findings paper.

## How to run
### Data preparation

1. Put the ASAP `train.tsv` and `test_public.tsv` files in `data/orig/ASAP`.
2. Put the SRA directories `/beetle/` and `/sciEntsBank/` in `data/orig/SRA`.
3. Run `python3 learning_curve/split_data.py`

You will then find the generated prompt-wise data split in `data/ASAP`, `data/SRA_2way` and `data/SRA_5way`.

### Experiments

Use the scripts in the RUN_* folders to run the different experiments, for example run `python3 RUN_BASELINE/ASAP_baseline.py` to use baseline methods to score the ASAP prompts.
You will then find the learning curve results in `results`.

If you wish to run the experiments in `RUN_CROSS`, be sure to first run `python3 RUN_CROSS/ASAP_train_base_models_BERT.py` and `python3 RUN_CROSS/ASAP_train_base_models_SBERT.py` to train the required base models.

### Result Analysis

Before running analysis scripts on results for the SRA corpus, run `python3 learning_curve/reorder_results.py` to adapt the data structure. You will then find dataset-separated results in `results/RESULTS_SEP_DATASETS`.
If you wish to only keep the learning curve stats, you can then generate a flatter representation of the results by running `python3 learning_curve/collect_results.py`, which will create a folder `results/RESULTS_REDUCED`.

To plot comparisons of the different methods, run `python3 learning_curve/aggregate_curves_with_majority_oracle.py`.
To determine which proportion of answers is only classified by one of the methods, run `python3 learning_curve/aggregate_curves_uniqueTo.py`.

## Citation
```
@inproceedings{bexte_similarity_2023,
    title = {Similarity-based content scoring - A more classroom-suitable alternative to instance-based scoring?},
    author = {Bexte, Marie and Horbach, Andrea and Zesch, Torsten},
    booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
    publisher = {Association for Computational Linguistics},
    location = {Toronto, Canada},
    date = {2023-07}
}
```
