from learning_curve.run_lc import run
import os
from copy import deepcopy

train_sizes=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
base_path="data/ASAP"

for prompt in range(1, 11):

    run(dataset_name="ASAP",
    prompt_name=str(prompt),
    train_path=os.path.join(base_path, str(prompt), "train.csv"),
    val_path=os.path.join(base_path, str(prompt), "val.csv"),
    test_path=os.path.join(base_path, str(prompt), "test.csv"),
    method="BERT",
    eval_measure="QWK",
    sampling_strategy="balanced",
    num_labels=5,
    upsample_training=True,
    predetermined_train_sizes=deepcopy(train_sizes))
