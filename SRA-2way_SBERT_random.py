from learning_curve.run_lc import run
import os
from copy import deepcopy

train_sizes=[2, 4, 6, 8, 10, 12, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]
dataset_name="SRA_2way"
base_path="data/SRA_2way"
eval_measure="weighted_f1"
sampling_strategy="random"
num_labels=2

with open('SRA_Beetle.txt', 'r') as prompts:
    
    for prompt in prompts:
        prompt=prompt.strip()
        
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
