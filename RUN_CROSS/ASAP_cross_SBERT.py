import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from learning_curve.run_lc_cross import run_cross
from copy import deepcopy
import pandas as pd

seed = 1234
num_train = 1250

train_sizes=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
base_path="data/ASAP"

bert_model_dir = "results/BASE_MODELS/SBERT/ASAP"

# Base model
for base_prompt in range(1, 11):

    target_prompts = list(range(1, 11))
    target_prompts.remove(base_prompt)

    for target_prompt in target_prompts:

        print(base_prompt, target_prompt)

        base_model_path = os.path.join(bert_model_dir, str(base_prompt), "finetuned_model")

        run_cross(results_folder=os.path.join("results_cross", "ASAP", "base_"+str(base_prompt)),
        dataset_name="ASAP",
        prompt_name=str(target_prompt),
        train_path=os.path.join(base_path, str(target_prompt), "train.csv"),
        val_path=os.path.join(base_path, str(target_prompt), "val.csv"),
        test_path=os.path.join(base_path, str(target_prompt), "test.csv"),
        method="SBERT",
        eval_measure="QWK",
        sampling_strategy="random",
        num_labels=5,
        upsample_training=True,
        predetermined_train_sizes=deepcopy(train_sizes),
        sbert_base=base_model_path)