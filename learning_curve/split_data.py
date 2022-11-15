from ast import Pass
from operator import sub
from random import random
import pandas as pd
import os
import sys
import xml.etree.ElementTree as ET


# Overall target folder, each dataset will become a subdir here
data_folder = "data"
# Desired order of columns in final files
data_columns = ["prompt", "id", "text", "label"]

# Random state to split data
state=4334


def has_at_least_two_labels(df):

    if len(df["label"].unique()) < 2:
        return False
    return True


def split_asap():

    val_size = 4

    # Read gold data
    asap_train_path = "/Users/mariebexte/Coding/Datasets/_content_scoring_datasets/en/ASAP/train.tsv"
    asap_test_path = "/Users/mariebexte/Coding/Datasets/_content_scoring_datasets/en/ASAP/test_public.txt"
    df_train = pd.read_csv(asap_train_path, sep="\t")
    df_test = pd.read_csv(asap_test_path, sep="\t")

    # Build target dir
    dataset_name = "ASAP"
    dataset_path = os.path.join(data_folder, dataset_name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Rename columns: Train
    df_train = df_train[["EssaySet", "Id", "EssayText", "Score1"]]
    df_train.columns = data_columns

    # Rename columns: Test
    df_test = df_test[["EssaySet", "Id", "EssayText", "essay_score"]]
    df_test.columns = data_columns


    ## Write data: Test (Just a matter of splitting gold testing data into prompts)
    # For each prompt
    for prompt in set(df_test["prompt"].unique()):

        # Make subdir
        prompt_path = os.path.join(dataset_path, str(prompt))
        if not os.path.exists(prompt_path):
            os.mkdir(prompt_path)
        
        # Save subset of answers to this prompt as file
        df_prompt = df_test[df_test["prompt"] == prompt]
        df_prompt.to_csv(os.path.join(prompt_path, "test.csv"), index=None)


    ## Write data: Train and val (Split away some of train to become val)
    # For each prompt
    for prompt in set(df_train["prompt"].unique()):

        # Make subdir if not present already
        prompt_path = os.path.join(dataset_path, str(prompt))
        if not os.path.exists(prompt_path):
            os.mkdir(prompt_path)

        # Take val_size instances of train as val
        df_prompt = df_train[df_train["prompt"] == prompt]
        df_val = df_prompt.sample(val_size, random_state=state)

        # If we end up with validation instances that all have the same label: Sample again
        num_try = 1
        while not has_at_least_two_labels(df_val):
            print("VAL IS NOT DIVERSE", prompt, df_val)
            df_val = df_prompt.sample(val_size, random_state=state+num_try)
            num_try += 1
        if num_try > 1:
            print(df_val)
        df_rest = df_prompt.drop(df_val.index)

        # Save val and remaining train as new training data
        df_val.to_csv(os.path.join(prompt_path, "val.csv"), index=None)
        df_rest.to_csv(os.path.join(prompt_path, "train.csv"), index=None)


# For SRA processing: Collect all XML files in dir into dictionary of dataframes
def get_df_dict(path):

    df_dict = {}

    # For each prompt
    for file in os.listdir(path):

        prompt = file[:file.index(".")]
        # Collect answers into dict first, later create dataframe from dict
        answer_dict = {}
        answer_index = 0

        # Collect all student answers into dataframe     
        root = ET.parse(os.path.join(path, file)).getroot()
        for answer in root.iter('studentAnswer'):
            answer_dict[answer_index] = {"prompt": prompt, "id": answer.get(
                "id"), "text": answer.text, "label": answer.get("accuracy")}
            answer_index += 1
        
        df_dict[prompt] = pd.DataFrame.from_dict(answer_dict, orient = "index")
    return df_dict


def write_SRA_testing(two_way_path, five_way_path, subset, test_path):

    # Beetle or SEB subfolders
    two_way_path_subset = os.path.join(two_way_path, subset)
    five_way_path_subset = os.path.join(five_way_path, subset)

    # Collect answers to all prompts into dataframes
    answers_path = os.path.join(test_path, "Core")
    answers_dict = get_df_dict(answers_path)

    # For each prompt
    for prompt in answers_dict:

        # Create subdir in 2- and 5-way dirs
        prompt_path_5way = os.path.join(five_way_path_subset, prompt)
        prompt_path_2way = os.path.join(two_way_path_subset, prompt)
        for path in [prompt_path_5way, prompt_path_2way]:
            if not os.path.exists(path):
                os.makedirs(path)

        df_answers = answers_dict[prompt]

        # Save prompt with 5-way annotations
        df_answers.to_csv(os.path.join(prompt_path_5way, os.path.basename(test_path) + ".csv"), index = None) 

        # Save prompt with 2-way annotations
        df_answers["label"][df_answers["label"] != "correct"] = "incorrect"
        df_answers.to_csv(os.path.join(prompt_path_2way, os.path.basename(test_path) + ".csv"), index = None)


def write_SRA_train_val(two_way_path, five_way_path, subset, train_path, val_size):

    # Beetle or SEB subfolders
    two_way_path_subset = os.path.join(two_way_path, subset)
    five_way_path_subset = os.path.join(five_way_path, subset)

    # Collect answers to all prompts into dataframes
    df_dict_train = get_df_dict(train_path)

    # For each prompt
    for prompt in df_dict_train.keys():

        # Create subdirs in 5- and 2-way directories
        prompt_path_5way = os.path.join(five_way_path_subset, prompt)
        prompt_path_2way = os.path.join(two_way_path_subset, prompt)
        for path in [prompt_path_5way, prompt_path_2way]:
            if not os.path.exists(path):
                os.makedirs(path)

        df_prompt = df_dict_train[prompt]

        # Sample val_size instances of the data for validation, rest becomes train
        df_val = df_prompt.sample(val_size, random_state=state)

        # If we end up with validation instances that all have the same label: Sample again
        num_try = 1
        while not has_at_least_two_labels(df_val):
            print("NOT DIVERSE", prompt, df_val)
            df_val = df_prompt.sample(val_size, random_state=state+num_try)
            num_try += 1
        if num_try > 1:
            print(df_val)
        df_train = df_prompt.drop(df_val.index)

        # Save prompt validation and testing answers with 5-way annotations
        df_train.to_csv(os.path.join(prompt_path_5way, "train.csv"), index = None)
        df_val.to_csv(os.path.join(prompt_path_5way, "val.csv"), index = None)

        # Map from 5- to 2-way labels
        df_train["label"][df_train["label"] != "correct"] = "incorrect"
        df_val["label"][df_val["label"] != "correct"] = "incorrect"

        # Save prompt validation and testing answers with 2-way annotations
        df_train.to_csv(os.path.join(prompt_path_2way, "train.csv"), index = None)
        df_val.to_csv(os.path.join(prompt_path_2way, "val.csv"), index = None)


def split_semEval():

    val_size = 4
    beetle_path = "/Users/mariebexte/Coding/Datasets/SemEval2013-Task7-5way/beetle"
    seb_path = "/Users/mariebexte/Coding/Datasets/SemEval2013-Task7-5way/sciEntsBank"

    dataset_name = "SRA"
    dataset_path_5way = os.path.join(data_folder, dataset_name + "_5way")
    dataset_path_2way = os.path.join(data_folder, dataset_name + "_2way")

    if not os.path.exists(dataset_path_5way):
        os.makedirs(dataset_path_5way)

    if not os.path.exists(dataset_path_2way):
        os.makedirs(dataset_path_2way)

    ## Process Beetle
    beetle_name = "Beetle"
    # Testing data
    for path in ["test-unseen-answers", "test-unseen-questions"]:
        write_SRA_testing(two_way_path=dataset_path_2way, five_way_path=dataset_path_5way, subset=beetle_name, test_path=os.path.join(beetle_path, path))
    # Training and validation data
    write_SRA_train_val(two_way_path=dataset_path_2way, five_way_path=dataset_path_5way, subset=beetle_name, train_path=os.path.join(beetle_path, "train", "Core"), val_size=val_size)

    ## Process SEB
    seb_name = "SEB"
    # Testing data
    for path in ["test-unseen-answers", "test-unseen-questions", "test-unseen-domains"]:
        write_SRA_testing(two_way_path=dataset_path_2way, five_way_path=dataset_path_5way, subset=seb_name, test_path=os.path.join(seb_path, path))
    # Training and validation data
    write_SRA_train_val(two_way_path=dataset_path_2way, five_way_path=dataset_path_5way, subset=seb_name, train_path=os.path.join(seb_path, "train", "Core"), val_size=val_size)


split_asap()
split_semEval()