from learning_curve.run_lc import run
import os
from copy import deepcopy

train_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
base_path="data/SRA_5way"

with open('SRA_Beetle.txt', 'r') as prompts:

    for prompt in prompts:

        prompt=prompt.strip()
        run(dataset_name="SRA-5way",
        prompt_name=str(prompt),
        train_path=os.path.join(base_path, str(prompt), "train.csv"),
        val_path=os.path.join(base_path, str(prompt), "val.csv"),
        test_path=os.path.join(base_path, str(prompt), "test-unseen-answers.csv"),
        method="BERT",
        eval_measure="weighted_f1",
        sampling_strategy="random",
        num_labels=5,
        predetermined_train_sizes=deepcopy(train_sizes))
