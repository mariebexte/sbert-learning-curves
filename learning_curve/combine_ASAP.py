from shutil import copyfile, copy
import pandas as pd
import os

# Aggregate results for longer ASAP learning curves
# target_loc = "FINAL_RESULTS/RESULTS_FULL_ASAP_LONG/ASAP/random"
target_loc = "results/FULL_ASAP_LONG/ASAP/random"

def copy(first_half_path, second_half_path, prompt, method, result_filename):

    first_half_results = os.path.join(first_half_path, str(prompt), method, result_filename)
    second_half_results = os.path.join(second_half_path, str(prompt), method, result_filename)

    if os.path.exists(first_half_results):
        first_half = pd.read_csv(first_half_results)#.reset_index()
    else:
        first_half = pd.DataFrame()

    if os.path.exists(second_half_results):
        second_half = pd.read_csv(second_half_results)#.reset_index()
    else:
        second_half = pd.DataFrame()

    # print(first_half)
    # print(second_half)

    full_results = pd.merge(first_half, second_half, how='left', left_index=True, right_index=True)

    target_path = os.path.join(target_loc, str(prompt), method)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    full_results.to_csv(os.path.join(target_path, result_filename), index=None)


def combine_methods(first_half_path, second_half_path, methods):

    for prompt in range(1, 11):

        for method in methods:

            copy(first_half_path=first_half_path, second_half_path=second_half_path, method=method, prompt=prompt, result_filename='QWK_lc_results.csv')

            if method in ["SBERT", "pretrained", "edit"]:
                copy(first_half_path=first_half_path, second_half_path=second_half_path, method=method, prompt=prompt, result_filename='QWK_lc_results_max.csv')


# first_half_path = "/Users/mariebexte/Coding/Projects/sbert-learning-curves/FINAL_RESULTS/TOTAL_RESULTS_REDUCED/ASAP/random"
# second_half_path = "/Users/mariebexte/Coding/Projects/sbert-learning-curves/FINAL_RESULTS/RESULTS_LONG_ASAP/ASAP/random"
# combine_methods(first_half_path=first_half_path, second_half_path=second_half_path, methods=["LR", "BERT", "SBERT", "RF", "SVM"])

# first_half_path = "FINAL_RESULTS/TOTAL_RESULTS_REDUCED/ASAP/random"
# second_half_path = "FINAL_RESULTS/RESULTS_LONG_ASAP_BASELINES/ASAP/random"
# combine_methods(first_half_path=first_half_path, second_half_path=second_half_path, methods=["edit", "pretrained"])

first_half_path = "results/RESULTS_REDUCED/ASAP/random"
second_half_path = "results_long/RESULTS_REDUCED/ASAP/random"
combine_methods(first_half_path=first_half_path, second_half_path=second_half_path, methods=["LR", "BERT", "SBERT", "RF", "SVM", "edit", "pretrained"])
