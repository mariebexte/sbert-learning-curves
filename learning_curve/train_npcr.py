from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import sys
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch


def prepare_df(df_ref, df_student, id_column="id", target_column="label", answer_column="text"):

    example_dict = {}
    example_index = 0

    for _, student_answer in df_student.iterrows():
        for _, reference_answer in df_ref.iterrows():

            if not reference_answer[id_column] == student_answer[id_column]:

                label = reference_answer[target_column] - student_answer[target_column]
                emb_diff = reference_answer["encoding"] - student_answer["encoding"]

                # print("EX 1", reference_answer["encoding"])
                # print("EX 2", student_answer["encoding"])
                # print("DIFF", emb_diff)

                example_dict[example_index] = {"ref_answer": reference_answer[answer_column], "ref_score": reference_answer[target_column], "student_answer": student_answer[answer_column], "student_answer_id": student_answer[id_column], "student_score": student_answer[target_column], "emb_diff": emb_diff, "score_diff": label}
                example_index += 1

    return pd.DataFrame.from_dict(example_dict, "index")


def train_npcr(df_train, df_test, df_val=None, method="LR", answer_column="text", target_column="label", id_column="id", sbert_base_model="all-MiniLM-L6-v2"):

    device = "cpu"
    #device = "mps"
    if torch.cuda.is_available():
        device = "cuda"

    model = SentenceTransformer(sbert_base_model, device=device)
    logging.info("Running NPCR using base model: "+str(sbert_base_model)+" !")

    # If there is validation data, merge it with the training data
    if df_val is not None:
        df_train = pd.concat([df_train, df_val])
        logging.info("SHALLOW: Combined training and validation.")
    else:
        logging.info("SHALLOW: There is no validation data, only training on train.")
    

    df_test.columns = df_train.columns

    if method == "LR":
        estimator = LogisticRegression(max_iter=1000)

    elif method == "RF":
        estimator = RandomForestClassifier()

    elif method == "SVM":
        estimator = SVC()

    else:
        logging.info("Unknown shallow learning method: "+method+"! Please pick one of the following: 'LR', 'RF', 'SVM'!")
        print("Unknown shallow learning method:", method)
        sys.exit(0)

    df_train = df_train.copy()
    df_test = df_test.copy()

    # Create embedding distances and prediction scores
    for df in [df_train, df_test]:
        df["encoding"] = df[answer_column].apply(model.encode)

    df_test = prepare_df(df_ref=df_train, df_student=df_test, id_column=id_column, target_column=target_column, answer_column=answer_column)
    df_train = prepare_df(df_ref=df_train, df_student=df_train, id_column=id_column, target_column=target_column, answer_column=answer_column)

    X_train = list(df_train["emb_diff"])
    X_test = list(df_test["emb_diff"])

    y_train = list(df_train["score_diff"])
    # y_test = df_test[target_col]

    try:
        model = estimator.fit(X_train, y_train)
        y_pred_diff = model.predict(X_test)
        df_test["pred_diff"] = y_pred_diff
        df_test["pred"] = df_test["ref_score"] - df_test["pred_diff"]
        # print(list(df_test.groupby(["student_answer_id"])))
        predictions_series = df_test.groupby('student_answer_id')['pred'].agg(lambda x: pd.Series.mode(x)[0])
        #predictions_series = df_test.groupby(['student_answer_id'])['pred'].agg(pd.Series.mode)
        #predictions_series = predictions_series.str[-1].fillna(predictions_series.mask(predictions_series.str.len().eq(0)))
        predictions = list(predictions_series)
        # print(predictions)

        return predictions

    except ValueError:
        if len(df_train[target_column].unique()) == 1:
            # Training has just one label, return this as prediction for all instances
            majority_label = df_train[target_column].mode()[0]
            logging.warn("All training examples have the same label: "+majority_label+"! Returning this as prediction for all testing instances!")
            return [majority_label] * len(df_test)
        else:
            print("A ValueError ocurred and was not due to the training data containing only one label.")
            logging.error("A ValueError ocurred and was not due to the training data containing only one label.")
            sys.exit(0)