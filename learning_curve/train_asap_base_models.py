import pandas as pd
import os
from learning_curve.train_bert import train_bert
from learning_curve.train_sbert_limited import train_sbert_limited
import sys
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, classification_report, f1_score
from learning_curve.run_lc_limited import write_classification_statistics

asap_base = "data/ASAP/"
seed = 1234
num_train = 1250

results_dir = "BASE_MODELS"

# For each prompt train a model with larg(est possible) amount of training data (same across all prompts)
# 1250 answers per prompt
# Sample in a way that keeps them for later experiments
# Save models to continue finetuning them later


def eval_predictions(run_path, df_train_subset, df_test, y_pred, y_pred_max, method, eval_measure="QWK", target_column="label"):

    # Calculate evaluation metrics
    y_true=list(df_test[target_column])

    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    weighted_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    # For SBERT: Also calculate QWK and weighted F1 for predictions obtained using max strategy
    if method in ["SBERT", "edit", "overlap", "pretrained", "cosine"]:
        qwk_max = cohen_kappa_score(y_true, y_pred_max, weights='quadratic')
        weighted_f1_max = f1_score(y_true=y_true, y_pred=y_pred_max, average='weighted')

    # Write classification statistics to file
    write_classification_statistics(filepath=os.path.join(run_path, "results.txt"), y_true=y_true, y_pred=y_pred, qwk=qwk)
    # For SBERT: Also write classification statistics for predictions obtained using max strategy
    if method in ["SBERT", "edit", "overlap", "pretrained", "cosine"]:
        write_classification_statistics(filepath=os.path.join(run_path, "results_max.txt"), y_true=y_true, y_pred=y_pred_max, qwk=qwk_max)

    # Write training instances to file
    df_train_subset.to_csv(os.path.join(run_path, "train.csv"), index=None)

    # Write predictions to file
    df_predictions = pd.DataFrame({'id': list(df_test["id"]), 'y_true': y_true, 'y_pred': y_pred})
    df_predictions.to_csv(os.path.join(run_path, "predictions.csv"), index=None)
    # For SBERT: Also write predictions obtained using max strategy
    if method in ["SBERT", "edit", "overlap", "pretrained", "cosine"]:
        df_predictions_max = pd.DataFrame({'id': list(df_test["id"]), 'y_true': y_true, 'y_pred': y_pred_max})
        df_predictions_max.to_csv(os.path.join(run_path, "predictions_max.csv"), index=None)



def train_base_models_bert():

    for prompt in range(1, 11):

        prompt_path = os.path.join(asap_base, str(prompt), "train.csv")
        train_all = pd.read_csv(prompt_path)

        train_sample = train_all.sample(num_train, random_state=seed)

        df_val = pd.read_csv(os.path.join(asap_base, str(prompt), "val.csv"))
        df_test = pd.read_csv(os.path.join(asap_base, str(prompt), "test.csv"))
        
        run_path = os.path.join(results_dir, "BERT", "ASAP", str(prompt))
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        y_pred = train_bert(run_path=run_path, df_train=train_sample, df_val=df_val, df_test=df_test, save_model=True)

        eval_predictions(run_path=run_path, df_train_subset=train_sample, df_test=df_test, y_pred=y_pred, y_pred_max=None, method="BERT")



def train_base_models_sbert():

    for prompt in range(1, 11):

        prompt_path = os.path.join(asap_base, str(prompt), "train.csv")
        train_all = pd.read_csv(prompt_path)

        train_sample = train_all.sample(num_train, random_state=seed)

        df_val = pd.read_csv(os.path.join(asap_base, str(prompt), "val.csv"))
        df_test = pd.read_csv(os.path.join(asap_base, str(prompt), "test.csv"))
        
        run_path = os.path.join(results_dir, "SBERT", "ASAP", str(prompt))
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        # Returns predictions obtained using max and avg (default) strategy
        y_pred_max, y_pred = train_sbert_limited(run_path=run_path, df_train=train_sample, df_val=df_val, df_test=df_test, save_model=True)

        eval_predictions(run_path=run_path, df_train_subset=train_sample, df_test=df_test, y_pred=y_pred, y_pred_max=y_pred_max, method="SBERT")
