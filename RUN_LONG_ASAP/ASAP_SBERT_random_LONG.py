import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from learning_curve.run_lc_limited import run_limited
import os
from copy import deepcopy

train_sizes=[100, 300, 500, 750, 1000, 1250, 1500, 1750]

dataset_name="ASAP"
base_path="data/ASAP"
eval_measure="QWK"
sampling_strategy="random"
num_labels=5

for prompt in range(1, 11):

    run_limited(dataset_name=dataset_name,
    prompt_name=str(prompt),
    train_path=os.path.join(base_path, str(prompt), "train.csv"),
    val_path=os.path.join(base_path, str(prompt), "val.csv"),
    test_path=os.path.join(base_path, str(prompt), "test.csv"),
    method="LR",
    eval_measure=eval_measure,
    sampling_strategy=sampling_strategy,
    num_labels=num_labels,
    upsample_training=True,
    predetermined_train_sizes=deepcopy(train_sizes))
    
    run_limited(dataset_name=dataset_name,
    prompt_name=str(prompt),
    train_path=os.path.join(base_path, str(prompt), "train.csv"),
    val_path=os.path.join(base_path, str(prompt), "val.csv"),
    test_path=os.path.join(base_path, str(prompt), "test.csv"),
    method="RF",
    eval_measure=eval_measure,
    sampling_strategy=sampling_strategy,
    num_labels=num_labels,
    upsample_training=True,
    predetermined_train_sizes=deepcopy(train_sizes))
    
    run_limited(dataset_name=dataset_name,
    prompt_name=str(prompt),
    train_path=os.path.join(base_path, str(prompt), "train.csv"),
    val_path=os.path.join(base_path, str(prompt), "val.csv"),
    test_path=os.path.join(base_path, str(prompt), "test.csv"),
    method="SVM",
    eval_measure=eval_measure,
    sampling_strategy=sampling_strategy,
    num_labels=num_labels,
    upsample_training=True,
    predetermined_train_sizes=deepcopy(train_sizes))
    
    run_limited(dataset_name=dataset_name,
    prompt_name=str(prompt),
    train_path=os.path.join(base_path, str(prompt), "train.csv"),
    val_path=os.path.join(base_path, str(prompt), "val.csv"),
    test_path=os.path.join(base_path, str(prompt), "test.csv"),
    method="SBERT",
    eval_measure=eval_measure,
    sampling_strategy=sampling_strategy,
    num_labels=num_labels,
    upsample_training=True,
    predetermined_train_sizes=deepcopy(train_sizes))
