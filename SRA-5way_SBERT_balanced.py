from learning_curve.run_lc import run
import os
from copy import deepcopy

train_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
dataset_name="SRA_5way"
base_path="data/SRA_5way"
eval_measure="weighted_f1"
sampling_strategy="balanced"
num_labels=5

prompt_list = []

with open('SRA_Beetle.txt', 'r') as prompts:

    for prompt in prompts:

        # Prompts are ordered from having most to least answers
        prompt = prompt.strip()
        prompt_list.insert(0, prompt)


for prompt in prompt_list:

    run(dataset_name=dataset_name,
    prompt_name=str(prompt),
    train_path=os.path.join(base_path, str(prompt), "train.csv"),
    val_path=os.path.join(base_path, str(prompt), "val.csv"),
    test_path=os.path.join(base_path, str(prompt), "test-unseen-answers.csv"),
    method="LR",
    eval_measure=eval_measure,
    sampling_strategy=sampling_strategy,
    num_labels=num_labels,
    predetermined_train_sizes=deepcopy(train_sizes))
    
    run(dataset_name=dataset_name,
    prompt_name=str(prompt),
    train_path=os.path.join(base_path, str(prompt), "train.csv"),
    val_path=os.path.join(base_path, str(prompt), "val.csv"),
    test_path=os.path.join(base_path, str(prompt), "test-unseen-answers.csv"),
    method="RF",
    eval_measure=eval_measure,
    sampling_strategy=sampling_strategy,
    num_labels=num_labels,
    predetermined_train_sizes=deepcopy(train_sizes))
    
    run(dataset_name=dataset_name,
    prompt_name=str(prompt),
    train_path=os.path.join(base_path, str(prompt), "train.csv"),
    val_path=os.path.join(base_path, str(prompt), "val.csv"),
    test_path=os.path.join(base_path, str(prompt), "test-unseen-answers.csv"),
    method="SVM",
    eval_measure=eval_measure,
    sampling_strategy=sampling_strategy,
    num_labels=num_labels,
    predetermined_train_sizes=deepcopy(train_sizes))
    
    run(dataset_name=dataset_name,
    prompt_name=str(prompt),
    train_path=os.path.join(base_path, str(prompt), "train.csv"),
    val_path=os.path.join(base_path, str(prompt), "val.csv"),
    test_path=os.path.join(base_path, str(prompt), "test-unseen-answers.csv"),
    method="SBERT",
    eval_measure=eval_measure,
    sampling_strategy=sampling_strategy,
    num_labels=num_labels,
    predetermined_train_sizes=deepcopy(train_sizes))
