from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import sys
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import nltk
import pandas as pd
import numpy as np
from adapt.feature_based import FA
nltk.download('punkt')


def train_shallow_cross(method, df_cross, df_train, df_test, df_val=None, answer_column="text", target_column="label", cross_method='merge'):

    if cross_method == 'merge':
        return train_shallow_cross_merge_target_features(method, df_cross, df_train, df_test, df_val=df_val, answer_column="text", target_column="label")
    elif cross_method == 'easyadapt':
        return train_shallow_cross_easyadapt_limit_source_features(method, df_cross, df_train, df_test, df_val=df_val, answer_column="text", target_column="label")
    else:
        print("Unknown domain adaptation method for shallow learning: "+cross_method)
        logging.info("Unknown domain adaptation method for shallow learning: "+cross_method)


def train_shallow_cross_merge_target_features(method, df_cross, df_train, df_test, df_val=None, answer_column="text", target_column="label"):

    # If there is validation data, merge it with the training data
    if df_val is not None:
        df_train = pd.concat([df_train, df_val])
        logging.info("SHALLOW: Combined training and validation.")
    else:
        logging.info("SHALLOW: There is no validation data, only training on train.")

    # Add cross data into training data
    df_train_all = pd.concat([df_train, df_cross])
    logging.info("Added cross-prompt data into training.")

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

    X_train_discard = feature_extractor.fit_transform(list(df_train[answer_column]))
    X_train = feature_extractor.transform(list(df_train_all[answer_column]))
    X_test = feature_extractor.transform(list(df_test[answer_column]))

    y_train = list(df_train_all[target_column])
    # y_test = df_test[target_col]

    try:
        model = estimator.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    except ValueError:
        if len(df_train_all[target_column].unique()) == 1:
            # Training has just one label, return this as prediction for all instances
            majority_label = df_train_all[target_column].mode()[0]
            logging.warn("All training examples have the same label: "+majority_label+"! Returning this as prediction for all testing instances!")
            return [majority_label] * len(df_test)
        else:
            print("A ValueError ocurred and was not due to the training data containing only one label.")
            logging.error("A ValueError ocurred and was not due to the training data containing only one label.")
            sys.exit(0)


# Standard easyadapt: Determine features based on combined source and target inputs
def train_shallow_cross_easyadapt(method, df_cross, df_train, df_test, df_val=None, answer_column="text", target_column="label"):

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

    
    # First part: target data, second part: source data
    df_total = pd.concat([df_train, df_cross])
    # Jointly process into features (for consistent feature set across both domains)
    X_total = feature_extractor.fit_transform(list(df_total[answer_column])).toarray()
    # Cut first half: target features, rest: source features
    X_train_target = X_total[:len(df_train), :]
    X_train_source = X_total[len(df_train):, :]
    # print(len(df_train), X_train_target.shape)
    # print(len(df_cross), X_train_source.shape)

    #X_train_source = feature_extractor.fit_transform(list(df_cross[answer_column])).toarray()
    y_train_source = list(df_cross[target_column])

    #X_train_target = feature_extractor.fit_transform(list(df_train[answer_column])).toarray()
    y_train_target = list(df_train[target_column])
    X_test_target = feature_extractor.transform(list(df_test[answer_column])).toarray()
    # y_test_target = df_test[target_col]

    model = FA(estimator, Xt=X_train_target, yt=y_train_target, random_state=0)

    try:
        model.fit(X_train_source, y_train_source)
        y_pred = model.predict(X_test_target)
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


# Easyadapt with a feature set based on the target data
def train_shallow_cross_easyadapt_target_features(method, df_cross, df_train, df_test, df_val=None, answer_column="text", target_column="label"):

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

    
    # First part: target data, second part: source data
    #df_total = pd.concat([df_train, df_cross])
    # Jointly process into features (for consistent feature set across both domains)
    #X_total = feature_extractor.fit_transform(list(df_total[answer_column])).toarray()
    # Cut first half: target features, rest: source features
    X_train_target = feature_extractor.fit_transform(list(df_train[answer_column])).toarray()
    X_train_source = feature_extractor.transform(list(df_cross[answer_column])).toarray()
    # print(len(df_train), X_train_target.shape)
    # print(len(df_cross), X_train_source.shape)

    #X_train_source = feature_extractor.fit_transform(list(df_cross[answer_column])).toarray()
    y_train_source = list(df_cross[target_column])

    #X_train_target = feature_extractor.fit_transform(list(df_train[answer_column])).toarray()
    y_train_target = list(df_train[target_column])
    X_test_target = feature_extractor.transform(list(df_test[answer_column])).toarray()
    # y_test_target = df_test[target_col]

    model = FA(estimator, Xt=X_train_target, yt=y_train_target, random_state=0)

    try:
        model.fit(X_train_source, y_train_source)
        y_pred = model.predict(X_test_target)
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


# Easyadapt with 10000 most frequent source and all target ngrams
def train_shallow_cross_easyadapt_limit_source_features(method, df_cross, df_train, df_test, df_val=None, answer_column="text", target_column="label"):

    # If there is validation data, merge it with the training data
    if df_val is not None:
        df_train = pd.concat([df_train, df_val])
        logging.info("SHALLOW: Combined training and validation.")
    else:
        logging.info("SHALLOW: There is no validation data, only training on train.")

    # Shuffle training
    df_train = df_train.sample(frac=1, random_state=4334)

    df_test.columns = df_train.columns

    feature_extractor_source = CountVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        lowercase=True,
        tokenizer=word_tokenize)

    feature_extractor_target = CountVectorizer(
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

    
    # First part: target data, second part: source data
    #df_total = pd.concat([df_train, df_cross])
    # Jointly process into features (for consistent feature set across both domains)
    #X_total = feature_extractor.fit_transform(list(df_total[answer_column])).toarray()
    # Cut first half: target features, rest: source features
    X_train_target_targetSet = feature_extractor_target.fit_transform(list(df_train[answer_column])).toarray()
    X_train_source_sourceSet = feature_extractor_source.fit_transform(list(df_cross[answer_column])).toarray()

    X_train_target_sourceSet = feature_extractor_source.transform(list(df_train[answer_column])).toarray()
    X_train_source_targetSet = feature_extractor_target.transform(list(df_cross[answer_column])).toarray()

    X_train_source = np.concatenate((X_train_source_sourceSet, X_train_source_targetSet), axis=1)
    X_train_target = np.concatenate((X_train_target_sourceSet, X_train_target_targetSet), axis=1)

    #X_train_source = feature_extractor.fit_transform(list(df_cross[answer_column])).toarray()
    y_train_source = list(df_cross[target_column])

    #X_train_target = feature_extractor.fit_transform(list(df_train[answer_column])).toarray()
    y_train_target = list(df_train[target_column])

    X_test_target_targetSet = feature_extractor_target.transform(list(df_test[answer_column])).toarray()
    X_test_target_sourceSet = feature_extractor_source.transform(list(df_test[answer_column])).toarray()
    X_test_target = np.concatenate((X_test_target_sourceSet, X_test_target_targetSet), axis=1)
    # y_test_target = df_test[target_col]

    model = FA(estimator, Xt=X_train_target, yt=y_train_target, random_state=0)

    try:
        model.fit(X_train_source, y_train_source)
        y_pred = model.predict(X_test_target)
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