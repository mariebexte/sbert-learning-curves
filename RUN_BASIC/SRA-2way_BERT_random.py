import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from learning_curve.run_lc import run
from copy import deepcopy

train_sizes=[2, 4, 6, 8, 10, 12, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]
base_path="data/SRA_2way"

with open('SRA_Beetle.txt', 'r') as prompts:

    for prompt in prompts:
        prompt=prompt.strip()

        run(dataset_name="SRA_2way",
        prompt_name=str(prompt),
        train_path=os.path.join(base_path, str(prompt), "train.csv"),
        val_path=os.path.join(base_path, str(prompt), "val.csv"),
        test_path=os.path.join(base_path, str(prompt), "test-unseen-answers.csv"),
        method="BERT",
        eval_measure="weighted_f1",
        sampling_strategy="random",
        num_labels=2,
        predetermined_train_sizes=deepcopy(train_sizes))
