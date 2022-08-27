from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import sys
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import nltk
import pandas as pd
nltk.download('punkt')


def train_shallow(method, df_train, df_test, df_val=None, answer_column="text", target_column="label"):

    # If there is validation data, merge it with the training data
    if df_val is not None:
        df_train = pd.concat([df_train, df_val])
        logging.info("SHALLOW: Combined training and validation.")
    else:
        logging.info("SHALLOW: There is no validation data, only training on train.")
    
    # Shuffle training
    df_train = df_train.sample(frac=1, random_state=4334)

    df_test.columns = df_train.columns

    feature_extractor = CountVectorizer(
        ngram_range=(1, 3),
        lowercase=True,
        tokenizer=word_tokenize)

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

    X_train = feature_extractor.fit_transform(list(df_train[answer_column]))
    X_test = feature_extractor.transform(list(df_test[answer_column]))

    y_train = list(df_train[target_column])
    # y_test = df_test[target_col]

    try:
        model = estimator.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

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