import os
from learning_curve.run_lc_cross_shallow import run_cross_shallow
import os
from copy import deepcopy
import pandas as pd

asap_base = "data/ASAP/"
seed = 1234
num_train = 1250

train_sizes=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
base_path="data/ASAP"

# Shallow: Either just merge cross and within-prompt items into one bag of training examples ('merge')
# Or combine features: 'easyadapt'

def cross_shallow():

    for base_prompt in range(1, 11):

        train_path = os.path.join(asap_base, str(base_prompt), 'train.csv')
        df_train = pd.read_csv(train_path)
        train_sample = df_train.sample(num_train, random_state=seed)

        target_prompts = list(range(1, 11))
        target_prompts.remove(base_prompt)

        for shallow_cross_method in ['merge', 'easyadapt']:

            for target_prompt in target_prompts:

                #for shallow_method in ['LR', 'SVM', 'RF']:
                for shallow_method in ['LR']:

                    print(base_prompt, target_prompt)

                    run_cross_shallow(results_folder=os.path.join("results_cross", "ASAP", "base_"+str(base_prompt)),
                    dataset_name="ASAP",
                    prompt_name=str(target_prompt),
                    train_path=os.path.join(base_path, str(target_prompt), "train.csv"),
                    val_path=os.path.join(base_path, str(target_prompt), "val.csv"),
                    test_path=os.path.join(base_path, str(target_prompt), "test.csv"),
                    method=shallow_method,
                    eval_measure="QWK",
                    sampling_strategy="random",
                    num_labels=5,
                    upsample_training=True,
                    predetermined_train_sizes=deepcopy(train_sizes),
                    shallow_cross_method=shallow_cross_method,
                    shallow_cross_data=train_sample)
