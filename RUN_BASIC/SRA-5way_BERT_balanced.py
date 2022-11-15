import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from learning_curve.run_lc import run
from copy import deepcopy

train_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
base_path="data/SRA_5way"
prompt_list = []

with open('SRA_Beetle.txt', 'r') as prompts:

    for prompt in prompts:

        # Prompts are ordered from having most to least answers
        prompt = prompt.strip()
        prompt_list.insert(0, prompt)


for prompt in prompt_list:

    run(dataset_name="SRA_5way",
    prompt_name=str(prompt),
    train_path=os.path.join(base_path, str(prompt), "train.csv"),
    val_path=os.path.join(base_path, str(prompt), "val.csv"),
    test_path=os.path.join(base_path, str(prompt), "test-unseen-answers.csv"),
    method="BERT",
    eval_measure="weighted_f1",
    sampling_strategy="balanced",
    num_labels=5,
    predetermined_train_sizes=deepcopy(train_sizes))
