import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from copy import deepcopy
from sklearn.metrics import cohen_kappa_score, f1_score
from collections import Counter
from itertools import groupby

# Predefined colors for the different methods for ease of comparison across curves
colors = {"SBERT_max": "pink", "SBERT": "red", "LR": "blue", "RF": "green", "SVM": "orange", "BERT": "purple",
          "edit": "darkgoldenrod", "overlap": "black", "cosine": "lime", "pretrained": "deepskyblue", 'oracle': 'black', "all": "brown", "majority": "lime"}
# Predefined line styles for the different sampling strategies          
strategies = {"balanced": "-", "random": '--', "avg": "-", "max": "--"}

# For SRA labels: Map to integers
LABELS_TO_ID = {"correct": 5, "partially_correct_incomplete": 4, "irrelevant": 2, "incorrect": 3, "non_domain": 1, "contradictory": 0}
ID_TO_LABELS = {LABELS_TO_ID[key]: key for key in LABELS_TO_ID.keys()}

# EDIT HERE TO ADAPT
methods_to_consider = ["LR", "BERT", "SBERT", "edit", "pretrained"]

id_column = "id"
label_column = "label"

# Method to average a dataframe while respecting the requirements of the respective metric, i.e. Fisher transforming in case of QWK
def get_average(df, eval_metric):

    if eval_metric.lower() == "qwk":
        # Arctanh == FISHER
        df_preds_fisher = np.arctanh(df)
        test_scores_mean_fisher = np.nanmean(df_preds_fisher, axis=0)
        # Tanh == FISHERINV
        test_scores_mean = np.tanh(test_scores_mean_fisher)

    else:
        test_scores_mean = np.nanmean(df, axis=0)

    return test_scores_mean


def get_oracle(row):

    # Assign closest prediction everywhere (can either be correct or the 'least off')
    gold = row[label_column]

    initial_gold_label = gold
    gold = LABELS_TO_ID.get(gold, gold)
    label_is_string = gold!=initial_gold_label

    oracle_pred = -1
    diff = 1000000 # :)

    for method in methods_to_consider:

        try:
            pred = row[method]
            # For SRA, look up integer equivalent, if there is none, leave label unchanged
            pred = LABELS_TO_ID.get(pred, pred)

            if abs(pred-gold) < diff:
                oracle_pred = pred
                diff = abs(pred-gold)
        except:
            pass

    # Translate prediction back to string label (if needed)
    if label_is_string:
        oracle_pred = ID_TO_LABELS.get(oracle_pred, oracle_pred)
    return oracle_pred


def get_majority(row):

    votes=[]
    for method in methods_to_consider:
        try:
            votes.append(row[method])
        except:
            pass

    freqs = groupby(Counter(votes).most_common(), lambda x:x[1])

    # pick off the first group (highest frequency)
    vote = [val for val,count in next(freqs)[1]]
    # if(len(vote) > 1):
    #     print("MULTIPLES MAJORITY VOTES, TAKING FIRST:", vote, votes)
    # else:
    #     print("CLEAR MAJORITY:", vote, votes)

    return vote[0]


def eval_preds(y_true, y_pred, eval_measure):

    if eval_measure == "weighted_f1":
        return f1_score(y_true=y_true, y_pred=y_pred, average='weighted') 

    elif eval_measure.lower() == 'qwk':
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')

    else:
        print("Unknown evaluation measure:", eval_measure)
        sys.exit(0)


# Path to results for a certain prompts (expected to contain subdirs with results of the different methods)
def prompt_curve(data_path, result_path, dataset_name, sampling_strategy, eval_measure):

    marker = "SEP_DATASETS"
    subfolder = result_path[result_path.find(marker)+len(marker):result_path.find(sampling_strategy)].strip("/")

    prompt_name = os.path.basename(result_path)

    # Read in original test data
    try:
        df_test = pd.read_csv(os.path.join(data_path, prompt_name, "test.csv"))

    except:
        df_test = pd.read_csv(os.path.join(data_path, prompt_name, "test-unseen-answers.csv"))


    # Maps from training sizes to dicts with the different runs
    # For each run, there is a df (copy of testing, with columns indicating whether a certain method predicted a certain instance correctly)
    correct_ids_dict = {}

    for method in os.listdir(os.path.join(result_path)):
        if os.path.isdir(os.path.join(result_path, method)) and method in methods_to_consider:
          
            # For similarity-based results use max similarity
            if method in ["pretrained", "edit", "cosine", "overlap", "SBERT"]:
                # results_path = os.path.join(result_path, method, eval_measure+"_lc_results_max.csv")
                predictions_filename = "predictions_max.csv"
                method_name = method+"_max"
                
            else:
                # results_path = os.path.join(result_path, method, eval_measure+"_lc_results.csv")
                predictions_filename = "predictions.csv"
                method_name = method

            for training_size in os.listdir(os.path.join(result_path, method)):
                if os.path.isdir(os.path.join(result_path, method, training_size)):

                    for sample_run in os.listdir(os.path.join(result_path, method, str(training_size))):
                        if os.path.isdir(os.path.join(result_path, method, str(training_size), sample_run)):

                            # Read predictions
                            df_pred = pd.read_csv(os.path.join(result_path, method, str(training_size), sample_run, predictions_filename))
                            
                            training_size = int(training_size[training_size.rindex("_")+1:])

                            # Get dataframe for current run
                            training_size_dict = correct_ids_dict.get(training_size, {})
                            df_test_run = training_size_dict.get(sample_run, None)

                            # If this is the first method to report results for this run, initialize new copy
                            if df_test_run is None:
                                df_test_run = deepcopy(df_test)
                            
                            # Put predictions of this method in overall df
                            df_test_run[method] = df_pred['y_pred']

                            training_size_dict[sample_run] = df_test_run
                            correct_ids_dict[training_size] = training_size_dict


    # Dataframe to collect final data: Columns = Training sizes, Rows = Methods, Cells = Average fraction (over the n=20 runs) of test instances only this method classifies correctly
    # Rows are methods + all + oracle
    df_majority_oracle = None

    # Each training size becomes one column in final dataframe
    for training_size in correct_ids_dict:

        method_results = {}

        # Need to average the determined counts over however many sample runs there are
        for sample_run in correct_ids_dict[training_size]:

            # df_run has predictions of all methods for this specific run
            df_run = correct_ids_dict[training_size][sample_run]

            # add majority predictions
            df_run['majority'] = df_run.apply(get_majority, axis=1)

            # add oracle predictions
            df_run['oracle'] = df_run.apply(get_oracle, axis=1)

            # for each of the methods of interest, calculate performance (sanity check for the basic ones)
            for method in methods_to_consider + ['majority', 'oracle']:

                method_list = method_results.get(method, [])
                try:
                    method_list.append(eval_preds(y_true=df_run[label_column], y_pred=df_run[method], eval_measure=eval_measure))
                except:
                    # print("No predictions for method", method, "for training size", training_size, "and run", sample_run)
                    pass
                method_results[method] = method_list
         

        # Turn results into dataframe
        try:
            df_method_counts = pd.DataFrame.from_dict(method_results)
        except:
            # Not all methods have results for this training size
            print("Skipping results for training size", training_size, 'as not all methods have results for it')
            # print(method_results)
            continue
            
        # Average over the number of runs to get results for this training size
        df_train_size = df_method_counts.mean().to_frame(name=training_size).reset_index()
        df_train_size = df_train_size.rename(columns={'index': 'method'})

        # Append results for this training size to overall results
        if df_majority_oracle is None:
            df_majority_oracle = df_train_size
        else:
            df_majority_oracle = pd.merge(df_majority_oracle, df_train_size)

    print(df_majority_oracle)
    if df_majority_oracle is not None:

        # Rerder columns
        columns = list(df_majority_oracle.columns)
        columns.remove('method')
        columns.sort()
        df_majority_oracle = df_majority_oracle[['method']+columns]

        df_majority_oracle = df_majority_oracle.set_index('method')
        # df_majority_oracle["AVG"] = df_majority_oracle.mean(axis=1)

        for method, row in df_majority_oracle.iterrows():

            plt.plot(row, "o-", color=colors[method], label=method)

        df_majority_oracle = df_majority_oracle.reset_index()

    plt.title(dataset_name+": "+prompt_name+" ("+sampling_strategy+")", fontsize=12)
    # plt.ylim([0, 500])
    plt.ylim([0, 1])
    plt.xlabel("# of training examples", fontsize=20)
    plt.ylabel(eval_measure, fontsize=10)

    # Plot learning curve
    plt.grid()

    plt.rcParams['savefig.dpi'] = 300
    plt.rc('legend', fontsize=4, handlelength=1)
    plt.legend(loc="lower right")
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(result_path, dataset_name+"_"+sampling_strategy+"_"+prompt_name+"_oraclePerf_method-comparison.png"))
 
    plt.clf()
    plt.cla()
    plt.close()


    # Return dataframe with average results for comparison of sampling strategies
    if df_majority_oracle is not None:
        df_majority_oracle.insert(0, 'prompt', prompt_name)
        df_majority_oracle.to_csv(os.path.join(result_path, dataset_name+"_"+sampling_strategy+"_"+prompt_name+"_oraclePerf_method-comparison.csv"), index=None)
    return df_majority_oracle


# Path to results obtained with a certain sampling strategy (expected to contain folders with results to different promots)
def strategy_curve(data_path, result_path, dataset_name, eval_measure):

    sampling_strategy = os.path.basename(result_path)

    # key: a method, value: a dataframe containing the per-prompt results for this method
    method_results_dict = {}

    # For each prompt: Sort method results into respective dataframe
    for prompt in os.listdir(result_path):
        if os.path.isdir(os.path.join(result_path, prompt)):

            prompt_results = prompt_curve(data_path=data_path, result_path=os.path.join(result_path, prompt), dataset_name=dataset_name, sampling_strategy=sampling_strategy, eval_measure=eval_measure)

            # Can be a prompt where no calculation of LC was possible, i.e. balanced sampling of a prompt where not all labels are present
            if prompt_results is not None and len(prompt_results) > 0:

                for method in prompt_results["method"]:

                    method_overview = method_results_dict.get(method, pd.DataFrame())
                    # Append to dataframe with results for this method
                    method_overview = pd.concat([method_overview, prompt_results[prompt_results["method"] == method]], axis=0, ignore_index=True)
                    method_results_dict[method] = method_overview
    

    ## Average results for the different methods (= from results for individual prompts to overall results)

    # Dataframe to track how many prompts went into the average for a certain training size
    # Rows: methods, Cols: train_sizes, Cells: number_of_prompts
    support_df = pd.DataFrame()

    # Dataframe listing average performance of the different methods
    # Rows: methods, Cols: train_sizes, Cells: avg_performance
    overview_df = pd.DataFrame()

    # For each method
    for method in method_results_dict:

        # Save overview dataframe for this metric
        method_overview_df = method_results_dict[method]
        method_overview_df.to_csv(os.path.join(result_path, dataset_name+"_"+sampling_strategy+"_"+method+"_oraclePerf_prompt-comparison.csv"), index=None)

        # Remove method and prompt info from dataframe before aggregating
        train_sizes = method_overview_df.columns.tolist()
        train_sizes.remove("method")
        train_sizes.remove("prompt")
        method_overview_df = method_overview_df[train_sizes]

        # Append to support df to keep track of how many prompts factor into each average
        method_support = method_overview_df.count()
        method_support_df = pd.DataFrame(method_support).transpose()
        # Add method information as first column
        method_support_df.insert(0, 'method', method)
        # Append information about current method to overall support dataframe
        support_df = pd.concat([support_df, method_support_df])

        # Calculate per-method average: Plot and collect into overview df
        method_avg = get_average(method_overview_df, eval_measure)
        plt.plot(method_overview_df.columns, method_avg, "o-", color=colors[method], label=method)
        method_avg_df = pd.DataFrame([method_avg], columns=method_overview_df.columns)
        method_avg_df.insert(0, 'method', method)
        overview_df = pd.concat([overview_df, method_avg_df])

    # Save support df to csv
    # Where there are no values: Support is 0
    support_df = support_df.fillna(0)
    support_df.to_csv(os.path.join(result_path, dataset_name+"_"+sampling_strategy+"_oraclePerf_support.csv"), index=None)

    # Save method overview df to csv
    overview_df.to_csv(os.path.join(result_path, dataset_name+"_"+sampling_strategy+"_oraclePerf_method-comparison.csv"),index=None)

    plt.title(dataset_name+" ("+sampling_strategy+")", fontsize=12)
    plt.ylim([0, 1])
    plt.xlabel("# of training examples", fontsize=20)
    plt.ylabel(eval_measure, fontsize=10)

    plt.grid()

    plt.rcParams['savefig.dpi'] = 300
    plt.rc('legend', fontsize=4, handlelength=1)
    plt.legend(loc="lower right")
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(result_path, dataset_name+"_"+sampling_strategy+"_oraclePerf_method-comparison.png"))

    plt.clf()
    plt.cla()
    plt.close()

    # Return overview dataframe for plotting comparison between sampling strategies
    return overview_df


# Path to the results of an entire dataset (expected to contain subdirs with sampling strategy results)
def dataset_curve(data_path, result_path, eval_measure):

    if not os.path.exists(result_path):
        print("Dir", data_path, "does not exist, skipping.")
        return None

    dataset_name = os.path.basename(result_path)
    # Dataframe to collect overall averaged results per method and sampling strategy
    overall_results = pd.DataFrame()

    # For each sampling strategy: Get averaged results
    for sampling_strategy in os.listdir(result_path):
        if os.path.isdir(os.path.join(result_path, sampling_strategy)) and not sampling_strategy=="unique_lists":

            strategy_results = strategy_curve(data_path=data_path, result_path=os.path.join(result_path, sampling_strategy), dataset_name=dataset_name, eval_measure=eval_measure)

            if len(strategy_results) > 0:

                # For each method: Get averaged results for the different methods
                for method in strategy_results["method"]:
                    strategy_results_method = strategy_results[strategy_results["method"] == method]
                    # Insert information about sampling strategy into results dataframe
                    strategy_results_method.insert(1, "strategy", sampling_strategy)
                    # Add to overall results dataframe
                    overall_results = pd.concat([overall_results, strategy_results_method])


    # Save result overview dataframe to csv
    overall_results.to_csv(os.path.join(result_path, dataset_name+"_oraclePerf_strategy_comparison.csv"), index=None)

    # Reduce columns to just the training sizes for easier plotting
    train_sizes = overall_results.columns.tolist()
    if len(overall_results) > 0:
        train_sizes.remove("method")
        train_sizes.remove("strategy")

    # Plot the individual averaged results
    for _, row in overall_results.iterrows():

        method = row["method"]
        strategy = row["strategy"]
        row = row[train_sizes]
        plt.plot(train_sizes, row.tolist(), linestyle=strategies[strategy], color=colors[method], label=method+"_"+strategy)


    plt.title(dataset_name, fontsize=12)
    plt.ylim([0, 1])
    plt.xlabel("# of training examples", fontsize=20)
    plt.ylabel(eval_measure, fontsize=10)

    plt.grid(linewidth = 0.1)

    plt.rcParams['savefig.dpi'] = 300
    plt.rc('legend', fontsize=4, handlelength=3)
    plt.legend(loc="lower right")
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(result_path, dataset_name+"_oraclePerf_strategy_comparison.png"))

    plt.clf()
    plt.cla()
    plt.close()


dataset_curve(data_path="data/ASAP", result_path="results/ASAP", eval_measure="QWK")
dataset_curve(data_path="data/ASAP", result_path="results/FULL_ASAP_LONG/ASAP", eval_measure="QWK")

dataset_curve(data_path="data/SRA_2way/Beetle", result_path="results/SEP_DATASETS/Beetle/SRA_2way", eval_measure="weighted_f1")
dataset_curve(data_path="data/SRA_5way/Beetle", result_path="results/SEP_DATASETS/Beetle/SRA_5way", eval_measure="weighted_f1")

dataset_curve(data_path="data/SRA_2way/SEB", result_path="results/SEP_DATASETS/SEB/SRA_2way", eval_measure="weighted_f1")
dataset_curve(data_path="data/SRA_5way/SEB", result_path="results/SEP_DATASETS/SEB/SRA_5way", eval_measure="weighted_f1")