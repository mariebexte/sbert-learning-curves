import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# Predefined colors for the different methods for ease of comparison across curves
colors = {"SBERT_max": "pink", "SBERT": "red", "LR": "blue", "RF": "green", "SVM": "orange", "BERT": "purple",
          "edit": "darkgoldenrod", "overlap": "black", "cosine": "lime", "pretrained": "deepskyblue"}
# Predefined line styles for the different sampling strategies          
strategies = {"balanced": "-", "random": '--', "avg": "-", "max": "--"}

result_folder = "unique_lists"

# EDIT THIS TO ADAPT PLOT
methods_to_consider = ["pretrained", "edit", 'LR', 'SBERT', 'BERT']

# Method to determine whether none of the methods gets an answer right
def check_if_no_method_gets_it(row):

    noone_gets_it = True
    for method in methods_to_consider:
        if row[method] == True:
            noone_gets_it = False
    return noone_gets_it


# Method to determine whether just one of the methods gets an answer right
def check_which_methods_get_it(row):

    num_methods_who_get_it = 0
    method_that_got_it = ""
    for method in methods_to_consider:
        if row[method] == True:
            num_methods_who_get_it += 1
            method_that_got_it = method
    if num_methods_who_get_it == 0:
        return "-none-"
    elif num_methods_who_get_it > 1:
        return "-multiple-"
    else:
        return method_that_got_it


def get_average(df, eval_metric):

    return np.nanmean(df, axis=0)


# Path to results for a certain prompts (expected to contain subdirs with results of the different methods)
# Result_path: Base path of the current result folder (grab from input)
# Dataset_path: Base path of the data the results are based on (user input) / Needed to look up testing data
def prompt_curve(results_path, dataset_path, sampling_strategy, prompt_name, dataset_name, eval_measure, id_column="id"):

    prompt_dir = os.path.join(results_path, sampling_strategy, prompt_name)

    # Dataframe to collect average results of the different methods
    prompt_results = pd.DataFrame()

    # Maps from training sizes to dicts with the different runs
    # For each run, there is further dict, mapping to the different methods
    # For each method there is a set, consisting of the ids of the instances that this method is able to predict correctly
    correct_ids_dict = {}

    for method in os.listdir(os.path.join(prompt_dir)):
        if os.path.isdir(os.path.join(prompt_dir, method)) and method in methods_to_consider:
            
            if method in ["SBERT", "pretrained", "edit", "cosine", "overlap"]:
                method_results_path = os.path.join(prompt_dir, method, eval_measure+"_lc_results_max.csv")
            else:
                method_results_path = os.path.join(prompt_dir, method, eval_measure+"_lc_results.csv")
            if os.path.exists(results_path):
                df_results = pd.read_csv(method_results_path)

                # For each training size
                for training_size in df_results.columns:
                    sample_counter = 0
                    for sample_run in os.listdir(os.path.join(prompt_dir, method, "train_size_"+str(training_size))):
                        if os.path.isdir(os.path.join(prompt_dir, method, 'train_size_'+str(training_size), sample_run)):
                            sample_counter += 1
                            sim_pred_path = os.path.join(os.path.join(prompt_dir, method, 'train_size_'+str(training_size), sample_run, 'predictions_sim.csv'))
                            col_name_gold="y_true"
                            col_name_pred="y_pred"
                            if os.path.exists(sim_pred_path):
                                df_pred = pd.read_csv(sim_pred_path)
                                col_name_gold="label"
                                col_name_pred="pred_max"
                            else:
                                df_pred = pd.read_csv(os.path.join(os.path.join(prompt_dir, method, 'train_size_'+str(training_size), sample_run, 'predictions.csv')))
                            set_correct = set(df_pred[df_pred[col_name_gold] == df_pred[col_name_pred]][id_column])
                            training_size_dict = correct_ids_dict.get(training_size, {})
                            run_dict = training_size_dict.get(sample_run, {})
                            run_dict[method] = (set_correct, df_pred)

                            training_size_dict[sample_run] = run_dict
                            correct_ids_dict[training_size] = training_size_dict


    # Dataframe to collect final data: Columns = Training sizes, Rows = Methods, Cells = Average number (over the n=20 runs) of test instances only this method classifies correctly
    df_unique_to_methods = None

    # Points from method names to a list [F, T] where F (T) is the average similarity of incorrect (correct) classifications 
    average_similarities = {}
    num_measures = 0

    # Points from method names to dict containing frequency counts of labels
    label_dist_of_uniques = {}
    label_dist_of_correct = {}

    # Each training size becomes one column in final dataframe
    for training_size in correct_ids_dict:

        method_counts = {}

        # Need to average the determined counts over however many sample runs there are
        for sample_run in correct_ids_dict[training_size]:

            num_measures += 1

            folder_path = os.path.join(results_path, result_folder, dataset_name, sampling_strategy, prompt_name, training_size, sample_run)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            df_test_orig = pd.read_csv(os.path.join(dataset_path, prompt_name, "test.csv"))
            # df_test_orig = pd.read_csv(os.path.join(dataset_path, prompt_name, "test_unseen_answers.csv"))

            for method in methods_to_consider:

                method_set = correct_ids_dict[training_size][sample_run][method][0]
                method_df = correct_ids_dict[training_size][sample_run][method][1]

                other_methods = list(correct_ids_dict[training_size][sample_run].keys())
                other_methods.remove(method)

                other_ids = set()
                for other_method in other_methods:
                    other_ids.update(correct_ids_dict[training_size][sample_run][other_method][0])

                # Set with all answer ids only this method gets correctly
                diff = method_set.difference(other_ids)

                method_count = method_counts.get(method, 0)
                method_count += len(diff)
                method_counts[method] = method_count

                # In folder structure: Place df with answer df
                method_df_unique = method_df[method_df[id_column].isin(diff)]
                method_df_unique.to_csv(os.path.join(folder_path, '-'.join(methods_to_consider)+"_uniqueTo_"+method+".csv"), index=None)

                df_test_orig[method] = df_test_orig[id_column].isin(method_set)

                # If it is a similarity-based method: Calculate average similarity of incorrect/correct classifications
                if method in ["edit", "pretrained", "overlap"]:
                    avg_sim = average_similarities.get(method, [0, 0])
                    correct = method_df[method_df["label"]==method_df["pred_max"]]
                    incorrect = method_df.drop(correct.index)
                    avg_sim[0] = avg_sim[0] + incorrect["sim_score_max"].mean()
                    avg_sim[1] = avg_sim[1] + correct["sim_score_max"].mean()
                    average_similarities[method] = avg_sim

                freq_dist = label_dist_of_uniques.get(method, {})
                if freq_dist == {}:
                    freq_dist["method"] = method
                try:
                    value_counts = method_df_unique["label"].value_counts()
                except:
                    value_counts = method_df_unique["y_true"].value_counts()
                for key in value_counts.keys():
                    before = freq_dist.get(key, 0)
                    freq_dist[key] = before + value_counts[key]
                label_dist_of_uniques[method] = freq_dist


                freq_dist = label_dist_of_correct.get(method, {})
                try:
                    correct = method_df[method_df["label"]==method_df["pred_max"]]
                except:
                    correct = method_df[method_df["y_true"]==method_df["y_pred"]]
                if freq_dist == {}:
                    freq_dist["method"] = method
                try:
                    value_counts = correct["label"].value_counts()
                except:
                    value_counts = correct["y_true"].value_counts()
                for key in value_counts.keys():
                    before = freq_dist.get(key, 0)
                    freq_dist[key] = before + value_counts[key]
                label_dist_of_correct[method] = freq_dist


            df_test_orig["correct_by"] = df_test_orig.apply(check_which_methods_get_it, axis=1)
            df_test_orig.to_csv(os.path.join(folder_path, "-".join(methods_to_consider)+".csv"), index=None)   
            
            
        # Average method_counts over the number of runs
        for method in method_counts:
            method_counts[method] = method_counts[method]/len(correct_ids_dict[training_size].keys())

        df_train_size = pd.DataFrame.from_dict(method_counts, orient='index', columns=[training_size]).reset_index()
        df_train_size = df_train_size.rename(columns={'index': 'method'})

        if df_unique_to_methods is None:
            df_unique_to_methods = df_train_size
        else:
            df_unique_to_methods = pd.merge(df_unique_to_methods, df_train_size)

    with open(os.path.join(results_path, result_folder, dataset_name, sampling_strategy, "-".join(methods_to_consider)+"_average_simiarities.csv"), 'w') as avg_sims:
        avg_sims.write("Method\tAvg_Incorrect\tAvg_Correct\n")
        for key in average_similarities.keys():
            avg_sims.write(key+"\t"+str(average_similarities[key][0]/num_measures)+"\t"+str(average_similarities[key][1]/num_measures)+"\n")

    label_dist = pd.DataFrame.from_dict(label_dist_of_uniques, orient="index")
    label_dist.to_csv(os.path.join(results_path, result_folder, dataset_name, sampling_strategy, "-".join(methods_to_consider)+"_label_dist.csv"), index=None)

    label_dist_corr = pd.DataFrame.from_dict(label_dist_of_correct, orient="index")
    label_dist_corr.to_csv(os.path.join(results_path, result_folder, dataset_name,  sampling_strategy, "-".join(methods_to_consider)+"_label_dist_corr.csv"), index=None)

    if df_unique_to_methods is not None:
        df_unique_to_methods = df_unique_to_methods.set_index('method')
        for method, row in df_unique_to_methods.iterrows():

            plt.plot(row, "o-", color=colors[method], label=method)
        df_unique_to_methods = df_unique_to_methods.reset_index()

    plt.title(dataset_name+": "+prompt_name+" ("+sampling_strategy+")", fontsize=12)
    #plt.ylim([0, 1])
    plt.xlabel("# of training examples", fontsize=20)
    plt.ylabel("# of answers only this method gets right", fontsize=10)

    # Plot learning curve
    plt.grid()

    plt.rcParams['savefig.dpi'] = 300
    plt.rc('legend', fontsize=4, handlelength=1)
    plt.legend(loc="lower right")
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(prompt_dir, dataset_name+"_"+sampling_strategy+"_"+prompt_name+"_numUnique_method-comparison.png"))
 
    plt.clf()
    plt.cla()
    plt.close()

    prompt_results.to_csv(os.path.join(prompt_dir, dataset_name+"_"+sampling_strategy+"_"+prompt_name+"_numUnique_method-comparison.csv"), index=None)

    # Return dataframe with average results for comparison of sampling strategies
    if df_unique_to_methods is not None:
        df_unique_to_methods.insert(0, 'prompt', prompt_name)
    return df_unique_to_methods



# Path to results obtained with a certain sampling strategy (expected to contain folders with results to different promots)
def strategy_curve(results_path, dataset_folder, strategy, dataset_name, eval_measure):

    strategy_path = os.path.join(results_path, strategy)

    sampling_strategy = os.path.basename(strategy_path)

    # key: a method, value: a dataframe containing the per-prompt results for this method
    method_results_dict = {}

    # For each prompt: Sort method results into respective dataframe
    for prompt in os.listdir(strategy_path):
        if os.path.isdir(os.path.join(strategy_path, prompt)):

            prompt_results = prompt_curve(results_path=results_path, dataset_path=dataset_folder, sampling_strategy=strategy, prompt_name=prompt, dataset_name=dataset_name, eval_measure=eval_measure)

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
        method_overview_df.to_csv(os.path.join(strategy_path, dataset_name+"_"+sampling_strategy+"_"+method+"_numUnique_prompt-comparison.csv"), index=None)

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
    support_df.to_csv(os.path.join(strategy_path, dataset_name+"_"+sampling_strategy+"_numUnique_support.csv"), index=None)

    # Save method overview df to csv
    overview_df.to_csv(os.path.join(strategy_path, dataset_name+"_"+sampling_strategy+"_numUnique_method-comparison.csv"),index=None)

    plt.title(dataset_name+" ("+sampling_strategy+")", fontsize=12)
    #plt.ylim([0, 1])
    plt.xlabel("# of training examples", fontsize=20)
    plt.ylabel("avg. # of answers only this method gets right", fontsize=10)

    plt.grid()

    plt.rcParams['savefig.dpi'] = 300
    plt.rc('legend', fontsize=4, handlelength=1)
    plt.legend(loc="lower right")
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(strategy_path, dataset_name+"_"+sampling_strategy+"_numUnique_method-comparison.png"))

    plt.clf()
    plt.cla()
    plt.close()

    # Return overview dataframe for plotting comparison between sampling strategies
    return overview_df


# Path to the results of an entire dataset (expected to contain subdirs with sampling strategy results)
def dataset_curve(results_folder, dataset_path, eval_measure):

    dataset_name = os.path.basename(results_folder)
    # Dataframe to collect overall averaged results per method and sampling strategy
    overall_results = pd.DataFrame()

    # For each sampling strategy: Get averaged results
    for sampling_strategy in os.listdir(results_folder):
        if os.path.isdir(os.path.join(results_folder, sampling_strategy)) and not sampling_strategy==result_folder:

            strategy_results = strategy_curve(results_path=results_folder, dataset_folder=dataset_path, strategy=sampling_strategy, dataset_name=dataset_name, eval_measure=eval_measure)

            if len(strategy_results) > 0:

                # For each method: Get averaged results for the different methods
                for method in strategy_results["method"]:
                    strategy_results_method = strategy_results[strategy_results["method"] == method]
                    # Insert information about sampling strategy into results dataframe
                    strategy_results_method.insert(1, "strategy", sampling_strategy)
                    # Add to overall results dataframe
                    overall_results = pd.concat([overall_results, strategy_results_method])


    # Save result overview dataframe to csv
    overall_results.to_csv(os.path.join(results_folder, dataset_name+"_numUnique_strategy_comparison.csv"), index=None)

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
    #plt.ylim([0, 1])
    plt.xlabel("# of training examples", fontsize=20)
    plt.ylabel("avg. # of answers only this method gets right", fontsize=10)

    plt.grid(linewidth = 0.1)

    plt.rcParams['savefig.dpi'] = 300
    plt.rc('legend', fontsize=4, handlelength=3)
    plt.legend(loc="lower right")
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(results_folder, dataset_name+"_numUnique_strategy_comparison.png"))

    plt.clf()
    plt.cla()
    plt.close()


dataset_curve(results_folder="results/ASAP", dataset_path="data/ASAP", eval_measure="QWK")

dataset_curve("results/SEP_DATASETS/Beetle/SRA_2way", "weighted_f1")
dataset_curve("results/SEP_DATASETS/Beetle/SRA_5way", "weighted_f1")

dataset_curve("results/SEP_DATASETS/SEB/SRA_2way", "weighted_f1")
dataset_curve("results/SEP_DATASETS/SEB/SRA_5way", "weighted_f1")