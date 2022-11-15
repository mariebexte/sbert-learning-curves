import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# Predefined colors for the different methods for ease of comparison across curves
colors = {"BERT_old": "green", "BERT_new": "orange", "BERT_old_save2": "green", "BERT_new_save2": "blue", "SBERT_noval": "deepskyblue", "BERT_10e": "red", "BERT_oldConfig": "red", "SBERT_wrongVal": "lime", "SBERT_wrongVal_max": "deepskyblue", "SBERT_valFixed": "lime", "SBERT_max": "pink", "SBERT": "red", "LR": "blue", "RF": "green", "SVM": "orange", "BERT": "purple",
          "SBERT_10e": "purple", "SBERT_larger": "purple", "SBERT_smaller": "blue", "BERT_150": "red", "BERT_4": "purple", "SBERT_150": "lime", "SBERT_4": "deepskyblue",
          "edit": "darkgoldenrod", "overlap": "black", "cosine": "lime", "pretrained": "deepskyblue", 'oracle': 'black'}
# Predefined line styles for the different sampling strategies          
strategies = {"balanced": "-", "random": '--', "avg": "-", "max": "--"}


# Method to average a dataframe while respecting the requirements of the respective metric, i.e. Fisher transforming in case of QWK
def get_average(df, eval_metric):

    # if eval_metric.lower() == "qwk":

    #     # Arctanh == FISHER
    #     df_preds_fisher = np.arctanh(df)
    #     test_scores_mean_fisher = np.nanmean(df_preds_fisher, axis=0)
    #     # Tanh == FISHERINV
    #     test_scores_mean = np.tanh(test_scores_mean_fisher)

    # else:
    test_scores_mean = np.nanmean(df, axis=0)

    return test_scores_mean


# Path to results for a certain prompts (expected to contain subdirs with results of the different methods)
def prompt_curve(prompt_dir, dataset_name, sampling_strategy, eval_measure):

    # methods_to_consider = ["RF", "pretrained", "edit"]
    methods_to_consider = ["BERT", "LR", "SBERT", "edit"]
    # methods_to_consider = ["edit", "cosine"]

    prompt_name = os.path.basename(prompt_dir)
    # Dataframe to collect average results of the different methods
    prompt_results = pd.DataFrame()

    # Maps from training sizes to dicts with the different runs
    # For each run, there is further dict, mapping to the different methods
    # For each method there is a set, consisting of the ids of the instances that this method is able to predict correctly
    correct_ids_dict = {}
    total_test = {"1": 557, "2": 426, "3": 406, "4": 295, "5": 598, "6": 599, "7": 599, "8": 599, "9": 599, "10": 546}

    for method in os.listdir(os.path.join(prompt_dir)):
        if os.path.isdir(os.path.join(prompt_dir, method)) and method in methods_to_consider:
            
            if method in ["pretrained", "edit", "cosine", "overlap"]:
                results_path = os.path.join(prompt_dir, method, eval_measure+"_lc_results_max.csv")
            else:
                results_path = os.path.join(prompt_dir, method, eval_measure+"_lc_results.csv")
            if os.path.exists(results_path):
                df_results = pd.read_csv(results_path)

                # For each training size
                for training_size in df_results.columns:
                    sample_counter = 0
                    for sample_run in os.listdir(os.path.join(prompt_dir, method, "train_size_"+str(training_size))):
                        if os.path.isdir(os.path.join(prompt_dir, method, 'train_size_'+str(training_size), sample_run)):
                            sample_counter += 1

                            df_pred = pd.read_csv(os.path.join(os.path.join(prompt_dir, method, 'train_size_'+str(training_size), sample_run, 'predictions.csv')))
                            set_correct = set(df_pred[df_pred['y_true'] == df_pred['y_pred']]['id'])
                            training_size_dict = correct_ids_dict.get(training_size, {})
                            run_dict = training_size_dict.get(sample_run, {})
                            run_dict[method] = set_correct

                            training_size_dict[sample_run] = run_dict
                            correct_ids_dict[training_size] = training_size_dict


    baseline_path = os.path.join("BASELINE_RESULTS_SEP_DATASETS", dataset_name, sampling_strategy, str(prompt_name))
    for method in os.listdir(baseline_path):
        if os.path.isdir(os.path.join(baseline_path, method)) and method in methods_to_consider:
            
            if method in ["pretrained", "edit", "cosine", "overlap"]:
                results_path = os.path.join(baseline_path, method, eval_measure+"_lc_results_max.csv")
            else:
                results_path = os.path.join(baseline_path, method, eval_measure+"_lc_results.csv")
            if os.path.exists(results_path):
                df_results = pd.read_csv(results_path)

                # For each training size
                for training_size in df_results.columns:
                    sample_counter = 0
                    for sample_run in os.listdir(os.path.join(baseline_path, method, "train_size_"+str(training_size))):
                        if os.path.isdir(os.path.join(baseline_path, method, 'train_size_'+str(training_size), sample_run)):
                            sample_counter += 1

                            df_pred = pd.read_csv(os.path.join(baseline_path, method, 'train_size_'+str(training_size), sample_run, 'predictions.csv'))
                            set_correct = set(df_pred[df_pred['y_true'] == df_pred['y_pred']]['id'])
                            training_size_dict = correct_ids_dict.get(training_size, {})
                            run_dict = training_size_dict.get(sample_run, {})
                            run_dict[method] = set_correct

                            training_size_dict[sample_run] = run_dict
                            correct_ids_dict[training_size] = training_size_dict

            # if method == 'SBERT':
            #     results_path = os.path.join(prompt_dir, method, eval_measure+"_lc_results_max.csv")
            #     if os.path.exists(results_path):
            #         df_results = pd.read_csv(results_path)

            #         # For each training size
            #         for training_size in df_results.columns:
            #             sample_counter = 0
            #             for sample_run in os.listdir(os.path.join(prompt_dir, method, "train_size_"+str(training_size))):
            #                 if os.path.isdir(os.path.join(prompt_dir, method, 'train_size_'+str(training_size), sample_run)):
            #                     sample_counter += 1

            #                     df_pred = pd.read_csv(os.path.join(os.path.join(prompt_dir, method, 'train_size_'+str(training_size), sample_run, 'predictions.csv')))
            #                     set_correct = set(df_pred[df_pred['y_true'] == df_pred['y_pred']]['id'])
            #                     training_size_dict = correct_ids_dict.get(training_size, {})
            #                     run_dict = training_size_dict.get(sample_run, {})
            #                     run_dict["SBERT_max"] = set_correct

            #                     training_size_dict[sample_run] = run_dict
            #                     correct_ids_dict[training_size] = training_size_dict


    # Dataframe to collect final data: Columns = Training sizes, Rows = Methods, Cells = Average number (over the n=20 runs) of test instances only this method classifies correctly
    df_unique_to_methods = None

    # Each training size becomes one column in final dataframe
    for training_size in correct_ids_dict:

        method_counts = {}

        # Need to average the determined counts over however many sample runs there are
        for sample_run in correct_ids_dict[training_size]:

            # for method in ["LR", "RF", "SVM", "BERT", "SBERT"]:
            #for method in ["pretrained", "edit", "cosine", "overlap"]:
            # for method in ["LR", "RF", "SVM", "BERT", "SBERT", "pretrained", "edit", "cosine", "overlap"]:
            for method in methods_to_consider:
            #for method in correct_ids_dict[training_size][sample_run]:
                #print(method)
                method_set = correct_ids_dict[training_size][sample_run][method]
                method_count = method_counts.get(method, 0)
                method_count += len(method_set)
                method_counts[method] = method_count
            

            # Add in oracle condition
            all_methods = list(correct_ids_dict[training_size][sample_run].keys())
            oracle = set()
            for other_method in all_methods:
                oracle.update(correct_ids_dict[training_size][sample_run][other_method])

            method_oracle = method_counts.get('oracle', 0)
            method_oracle += len(oracle)
            method_counts['oracle'] = method_oracle
            
            
        # Average method_counts over the number of runs
        for method in method_counts:
            method_counts[method] = method_counts[method]/len(correct_ids_dict[training_size].keys())

        # Calculate accuracy instead of absolute counts
        for method in method_counts:
            method_counts[method] = method_counts[method]/total_test[prompt_name]

        df_train_size = pd.DataFrame.from_dict(method_counts, orient='index', columns=[training_size]).reset_index()
        df_train_size = df_train_size.rename(columns={'index': 'method'})

        if df_unique_to_methods is None:
            df_unique_to_methods = df_train_size
        else:
            df_unique_to_methods = pd.merge(df_unique_to_methods, df_train_size)
       
    
    if df_unique_to_methods is not None:
        df_unique_to_methods = df_unique_to_methods.set_index('method')
        for method, row in df_unique_to_methods.iterrows():

            plt.plot(row, "o-", color=colors[method], label=method)
        df_unique_to_methods = df_unique_to_methods.reset_index()

    plt.title(dataset_name+": "+prompt_name+" ("+sampling_strategy+")", fontsize=12)
    # plt.ylim([0, 500])
    plt.ylim([0, 1])
    plt.xlabel("# of training examples", fontsize=20)
    plt.ylabel("# of correctly classified answers", fontsize=10)

    # Plot learning curve
    plt.grid()

    plt.rcParams['savefig.dpi'] = 300
    plt.rc('legend', fontsize=4, handlelength=1)
    plt.legend(loc="lower right")
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(prompt_dir, dataset_name+"_"+sampling_strategy+"_"+prompt_name+"_oracle_method-comparison.png"))
 
    plt.clf()
    plt.cla()
    plt.close()

    prompt_results.to_csv(os.path.join(prompt_dir, dataset_name+"_"+sampling_strategy+"_"+prompt_name+"_oracle_method-comparison.csv"), index=None)

    # Return dataframe with average results for comparison of sampling strategies
    if df_unique_to_methods is not None:
        df_unique_to_methods.insert(0, 'prompt', prompt_name)
    return df_unique_to_methods



# Path to results obtained with a certain sampling strategy (expected to contain folders with results to different promots)
def strategy_curve(strategy_path, dataset_name, eval_measure):

    sampling_strategy = os.path.basename(strategy_path)

    # key: a method, value: a dataframe containing the per-prompt results for this method
    method_results_dict = {}

    # For each prompt: Sort method results into respective dataframe
    for prompt in os.listdir(strategy_path):
        if os.path.isdir(os.path.join(strategy_path, prompt)):

            prompt_results = prompt_curve(os.path.join(strategy_path, prompt), dataset_name, sampling_strategy, eval_measure)

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
        method_overview_df.to_csv(os.path.join(strategy_path, dataset_name+"_"+sampling_strategy+"_"+method+"_oracle_prompt-comparison.csv"), index=None)

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
    support_df.to_csv(os.path.join(strategy_path, dataset_name+"_"+sampling_strategy+"_oracle_support.csv"), index=None)

    # Save method overview df to csv
    overview_df.to_csv(os.path.join(strategy_path, dataset_name+"_"+sampling_strategy+"_oracle_method-comparison.csv"),index=None)

    plt.title(dataset_name+" ("+sampling_strategy+")", fontsize=12)
    plt.ylim([0, 500])
    plt.xlabel("# of training examples", fontsize=20)
    plt.ylabel("# of correctly classified answers", fontsize=10)

    plt.grid()

    plt.rcParams['savefig.dpi'] = 300
    plt.rc('legend', fontsize=4, handlelength=1)
    plt.legend(loc="lower right")
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(strategy_path, dataset_name+"_"+sampling_strategy+"_oracle_method-comparison.png"))

    plt.clf()
    plt.cla()
    plt.close()

    # Return overview dataframe for plotting comparison between sampling strategies
    return overview_df


# Path to the results of an entire dataset (expected to contain subdirs with sampling strategy results)
def dataset_curve(dataset_path, eval_measure):

    dataset_name = os.path.basename(dataset_path)
    # Dataframe to collect overall averaged results per method and sampling strategy
    overall_results = pd.DataFrame()

    # For each sampling strategy: Get averaged results
    for sampling_strategy in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, sampling_strategy)):

            strategy_results = strategy_curve(os.path.join(dataset_path, sampling_strategy), dataset_name, eval_measure)

            if len(strategy_results) > 0:

                # For each method: Get averaged results for the different methods
                for method in strategy_results["method"]:
                    strategy_results_method = strategy_results[strategy_results["method"] == method]
                    # Insert information about sampling strategy into results dataframe
                    strategy_results_method.insert(1, "strategy", sampling_strategy)
                    # Add to overall results dataframe
                    overall_results = pd.concat([overall_results, strategy_results_method])


    # Save result overview dataframe to csv
    overall_results.to_csv(os.path.join(dataset_path, dataset_name+"_oracle_strategy_comparison.csv"), index=None)

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
    plt.ylim([0, 500])
    plt.xlabel("# of training examples", fontsize=20)
    plt.ylabel("# of correctly classified answers", fontsize=10)

    plt.grid(linewidth = 0.1)

    plt.rcParams['savefig.dpi'] = 300
    plt.rc('legend', fontsize=4, handlelength=3)
    plt.legend(loc="lower right")
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(dataset_path, dataset_name+"_oracle_strategy_comparison.png"))

    plt.clf()
    plt.cla()
    plt.close()


# dataset_curve("RESULTS_PRELIM_EXP/fin_results/SRA-2way", "weighted_f1")
# dataset_curve("RESULTS_PRELIM_EXP/fin_results/SRA-5way", "weighted_f1")

#dataset_curve("FINAL_RESULTS/fin_results_SBERTmodelComparison/SRA-2way", "weighted_f1")
#dataset_curve("FINAL_RESULTS/fin_results_SBERTmodelComparison/SRA-5way", "weighted_f1")

#dataset_curve("FINAL_RESULTS/fin_results_SBERTepochComparison/SRA-2way", "weighted_f1")
#dataset_curve("FINAL_RESULTS/fin_results_SBERTepochComparison/SRA-5way", "weighted_f1")

#dataset_curve("FINAL_RESULTS/fin_results_ASAPval/ASAP", "QWK")

# dataset_curve("RESULTS_PRELIM_EXP/fin_results_SBERT_VAL-FIXED/SRA-2way", "weighted_f1")
# dataset_curve("RESULTS_PRELIM_EXP/fin_results_SBERT_VAL-FIXED/SRA-5way", "weighted_f1")

# dataset_curve("RESULTS_PRELIM_EXP/fin_results_BERT_COMPARE-CONFIG/SRA-2way", "weighted_f1")
# dataset_curve("RESULTS_PRELIM_EXP/fin_results_BERT_COMPARE-CONFIG/SRA-5way", "weighted_f1")

# dataset_curve("RESULTS_PRELIM_EXP_FIXED/compare-sbert-models/SRA-2way", "weighted_f1")
# dataset_curve("RESULTS_PRELIM_EXP_FIXED/compare-sbert-models/SRA-5way", "weighted_f1")

# dataset_curve("RESULTS_PRELIM_EXP_FIXED/compare-sbert-10e/SRA-2way", "weighted_f1")
# dataset_curve("RESULTS_PRELIM_EXP_FIXED/compare-sbert-10e/SRA-5way", "weighted_f1")

# dataset_curve("RESULTS_PRELIM_EXP_FIXED/compare-bert-10e/SRA-2way", "weighted_f1")
# dataset_curve("RESULTS_PRELIM_EXP_FIXED/compare-bert-10e/SRA-5way", "weighted_f1")

# dataset_curve("RESULTS_PRELIM_EXP_FIXED/compare-sbert-noval/SRA-2way", "weighted_f1")
# dataset_curve("RESULTS_PRELIM_EXP_FIXED/compare-sbert-noval/SRA-5way", "weighted_f1")

# dataset_curve("RESULTS_PRELIM_EXP_FIXED/example-prompts/SRA-5way", "weighted_f1")
# dataset_curve("RESULTS_PRELIM_EXP_FIXED/example-prompts/SRA-2way", "weighted_f1")

# dataset_curve("RESULTS_PRELIM_EXP_FIXED/bert-sanity-check/SRA-5way", "weighted_f1")
# dataset_curve("RESULTS_PRELIM_EXP_FIXED/bert-sanity-check/SRA-2way", "weighted_f1")

dataset_curve("EXP_RESULTS_SEP_DATASETS/ASAP", "QWK")
# dataset_curve("EXP_RESULTS_SEP_DATASETS/Beetle/SRA_2way", "weighted_f1")
# dataset_curve("EXP_RESULTS_SEP_DATASETS/Beetle/SRA_5way", "weighted_f1")
# Does not exist for BERT
# dataset_curve("EXP_RESULTS_SEP_DATASETS/SEB/SRA_2way", "weighted_f1")
# Does not exist for BERT
#dataset_curve("EXP_RESULTS_SEP_DATASETS/SEB/SRA_5way", "weighted_f1")

# dataset_curve("BASELINE_RESULTS_SEP_DATASETS/ASAP", "QWK")
# dataset_curve("BASELINE_RESULTS_SEP_DATASETS/Beetle/SRA_2way", "weighted_f1")
# dataset_curve("BASELINE_RESULTS_SEP_DATASETS/Beetle/SRA_5way", "weighted_f1")
# dataset_curve("BASELINE_RESULTS_SEP_DATASETS/SEB/SRA_2way", "weighted_f1")
# dataset_curve("BASELINE_RESULTS_SEP_DATASETS/SEB/SRA_5way", "weighted_f1")

# dataset_curve("EXP_RESULTS/SRA_5way", "weighted_f1")
# dataset_curve("EXP_RESULTS/SRA_2way", "weighted_f1")