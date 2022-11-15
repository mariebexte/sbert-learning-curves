import os
from learning_curve.run_lc_cross import run_cross
import os
from copy import deepcopy

train_sizes=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
base_path="data/ASAP"

# Start with neural models
# Must call run_lc, but with differing base models & training data

def cross_bert():

    bert_model_dir = "ASAP_BASE_MODELS/BERT/ASAP"

    # Base model
    for base_prompt in range(1, 11):

        target_prompts = list(range(1, 11))
        target_prompts.remove(base_prompt)

        for target_prompt in target_prompts:

            print(base_prompt, target_prompt)

            base_model_path = os.path.join(bert_model_dir, str(base_prompt), "best_model")

            run_cross(results_folder=os.path.join("results_cross", "ASAP", "base_"+str(base_prompt)),
            dataset_name="ASAP",
            prompt_name=str(target_prompt),
            train_path=os.path.join(base_path, str(target_prompt), "train.csv"),
            val_path=os.path.join(base_path, str(target_prompt), "val.csv"),
            test_path=os.path.join(base_path, str(target_prompt), "test.csv"),
            method="BERT",
            eval_measure="QWK",
            sampling_strategy="random",
            num_labels=5,
            upsample_training=True,
            predetermined_train_sizes=deepcopy(train_sizes),
            bert_base=base_model_path)


def cross_sbert():

    bert_model_dir = "ASAP_BASE_MODELS/SBERT/ASAP"

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


def cross_sbert_pretrained():

    bert_model_dir = "ASAP_BASE_MODELS/SBERT/ASAP"

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
            method="pretrained",
            eval_measure="QWK",
            sampling_strategy="random",
            num_labels=5,
            upsample_training=True,
            predetermined_train_sizes=deepcopy(train_sizes),
            sbert_base=base_model_path)
