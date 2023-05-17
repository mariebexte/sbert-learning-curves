import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from learning_curve.run_lc import run
from copy import deepcopy

train_sizes=[100, 300, 500, 750, 1000]
base_path="data/ASAP"

for method in ["edit", "pretrained", "overlap", "cosine"]:

    for strategy in ["balanced", "random"]:

        for prompt in range(1, 11):

            run(dataset_name="ASAP",
            prompt_name=str(prompt),
            train_path=os.path.join(base_path, str(prompt), "train.csv"),
            val_path=os.path.join(base_path, str(prompt), "val.csv"),
            test_path=os.path.join(base_path, str(prompt), "test.csv"),
            method=method,
            eval_measure="QWK",
            sampling_strategy=strategy,
            num_labels=5,
            upsample_training=True,
            predetermined_train_sizes=deepcopy(train_sizes))
