import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# Predefined colors for the different methods for ease of comparison across curves
colors = {"BERT_old": "green", "BERT_new": "orange", "BERT_old_save2": "green", "BERT_new_save2": "blue", "SBERT_noval": "deepskyblue", "BERT_10e": "red", "BERT_oldConfig": "red", "SBERT_wrongVal": "lime", "SBERT_wrongVal_max": "deepskyblue", "SBERT_valFixed": "lime", "SBERT_max": "pink", "SBERT": "red", "LR": "blue", "RF": "green", "SVM": "orange", "BERT": "purple",
          "SBERT_10e": "purple", "SBERT_larger": "purple", "SBERT_smaller": "blue", "BERT_150": "red", "BERT_4": "purple", "SBERT_150": "lime", "SBERT_4": "deepskyblue",
          "edit": "darkgoldenrod", "edit_max": "green", "overlap": "turquoise", "overlap_max": "turquoise", "cosine": "orange", "cosine_max": "orange", "pretrained": "deepskyblue", "pretrained_max": "deepskyblue"}
# Predefined line styles for the different sampling strategies          
strategies = {"BERT": "-", "LR": '-', "SBERT": "-", "edit": "-", "SBERT_max": "-", "edit_max": "--", "overlap_max": "--", "cosine_max": "--", "pretrained_max": "--"}


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


# Path to results for a certain prompts (expected to contain subdirs with results of the different methods)
def prompt_curve(prompt_dir, dataset_name, sampling_strategy, eval_measure):

    prompt_name = os.path.basename(prompt_dir)
    # Dataframe to collect average results of the different methods
    prompt_results = pd.DataFrame()

    for method in os.listdir(os.path.join(prompt_dir)):
        if os.path.isdir(os.path.join(prompt_dir, method)):

            results_path = os.path.join(prompt_dir, method, eval_measure+"_lc_results.csv")
            # Might be a prompt without results or where a method is not done yet, therefore check for existance of file
            if os.path.exists(results_path):

                if method in ["SBERT", "BERT", "LR"]:
                #if method in ["SBERT", "BERT", "LR", "edit"]:

                    # Get averaged results for the current method and prompt
                    method_results = pd.read_csv(results_path)
                    method_results_avg = get_average(method_results, eval_measure)

                    # Collect results into overall dataframe to save and return
                    method_results_avg_df = pd.DataFrame([method_results_avg], columns=method_results.columns)
                    # Add method and prompt columns to overview dataframe
                    method_results_avg_df.insert(0, 'method', method)
                    method_results_avg_df.insert(0, 'prompt', prompt_name)
                    prompt_results = pd.concat([prompt_results, method_results_avg_df], axis=0, ignore_index=True)

                    # Plot average performance of current method
                    plt.plot(method_results.columns, method_results_avg, "o-", color=colors[method], label=method)

                # For SBERT, also plot max
                # if method in ["SBERT", "edit", "overlap", "cosine", "pretrained"]:

                #     results_path_max = os.path.join(prompt_dir, method, eval_measure+"_lc_results_max.csv")
                #     if os.path.exists(results_path_max):

                #         # Get average results for this prompt
                #         method_results_max = pd.read_csv(results_path_max)
                #         method_results_max_avg = get_average(method_results_max, eval_measure)

                #         method = method+"_max"

                #         # Collect results into overall dataframe to save and return
                #         method_results_max_avg_df = pd.DataFrame([method_results_max_avg], columns=method_results_max.columns)
                #         # Add method and prompt columns to overall dataframe
                #         method_results_max_avg_df.insert(0, 'method', method)
                #         method_results_max_avg_df.insert(0, 'prompt', prompt_name)
                #         prompt_results = pd.concat([prompt_results, method_results_max_avg_df], axis=0, ignore_index=True)

                #         # Plot average performance
                #         plt.plot(method_results_max.columns, method_results_max_avg, "o", color=colors[method], linestyle=strategies[method], label=method)


    # plt.title(dataset_name+": "+prompt_name+" ("+sampling_strategy+")", fontsize=12)
    plt.ylim([0, 1])
    # plt.xlabel("# of training examples", fontsize=20)
    plt.xlabel("Number of manually graded answers", fontsize=20)
    # plt.ylabel(eval_measure, fontsize=20)
    plt.ylabel("Grading quality of the model", fontsize=20)

    # Plot learning curve
    plt.grid()

    plt.rcParams['savefig.dpi'] = 300
    # plt.rc('legend', fontsize=4, handlelength=1)
    # plt.legend(loc="lower right")
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(prompt_dir, dataset_name+"_"+sampling_strategy+"_"+prompt_name+"_method-comparison.png"))
 
    plt.clf()
    plt.cla()
    plt.close()

    prompt_results.to_csv(os.path.join(prompt_dir, dataset_name+"_"+sampling_strategy+"_"+prompt_name+"_method-comparison.csv"), index=None)

    # Return dataframe with average results for comparison of sampling strategies
    return prompt_results


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
            if len(prompt_results) > 0:

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
        method_overview_df.to_csv(os.path.join(strategy_path, dataset_name+"_"+sampling_strategy+"_"+method+"_prompt-comparison.csv"), index=None)

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
        plt.plot(method_overview_df.columns, method_avg, "o", color=colors[method], linestyle=strategies[method], label=method)
        method_avg_df = pd.DataFrame([method_avg], columns=method_overview_df.columns)
        method_avg_df.insert(0, 'method', method)
        overview_df = pd.concat([overview_df, method_avg_df])

    # Save support df to csv
    # Where there are no values: Support is 0
    support_df = support_df.fillna(0)
    support_df.to_csv(os.path.join(strategy_path, dataset_name+"_"+sampling_strategy+"_support.csv"), index=None)

    # Save method overview df to csv
    overview_df.to_csv(os.path.join(strategy_path, dataset_name+"_"+sampling_strategy+"_method-comparison.csv"),index=None)

    plt.title(dataset_name+" ("+sampling_strategy+")", fontsize=12)
    plt.ylim([0, 1])
    plt.xlabel("# of training examples", fontsize=20)
    plt.ylabel(eval_measure, fontsize=20)

    plt.grid()

    plt.rcParams['savefig.dpi'] = 300
    plt.rc('legend', fontsize=4, handlelength=5)
    plt.legend(loc="lower right")
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(strategy_path, dataset_name+"_"+sampling_strategy+"_method-comparison.png"))

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
    overall_results.to_csv(os.path.join(dataset_path, dataset_name+"_strategy_comparison.csv"), index=None)

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
        plt.plot(train_sizes, row.tolist(), linestyle=strategies[method], color=colors[method], label=method+"_"+strategy)


    plt.title(dataset_name, fontsize=12)
    plt.ylim([0, 1])
    plt.xlabel("# of training examples", fontsize=20)
    plt.ylabel(eval_measure, fontsize=20)

    plt.grid(linewidth = 0.1)

    plt.rcParams['savefig.dpi'] = 300
    plt.rc('legend', fontsize=4, handlelength=3)
    plt.legend(loc="lower right")
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(dataset_path, dataset_name+"_strategy_comparison.png"))

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

# dataset_curve("EXP_RESULTS_SEP_DATASETS/ASAP", "QWK")
# dataset_curve("EXP_RESULTS_SEP_DATASETS/Beetle/SRA_2way", "weighted_f1")
# dataset_curve("EXP_RESULTS_SEP_DATASETS/Beetle/SRA_5way", "weighted_f1")
# dataset_curve("EXP_RESULTS_SEP_DATASETS/SEB/SRA_2way", "weighted_f1")
# dataset_curve("EXP_RESULTS_SEP_DATASETS/SEB/SRA_5way", "weighted_f1")
# dataset_curve("BASELINE_RESULTS_SEP_DATASETS/ASAP", "QWK")
# dataset_curve("BASELINE_RESULTS_SEP_DATASETS/SEB/SRA_5way", "weighted_f1")
# dataset_curve("BASELINE_RESULTS_SEP_DATASETS/SEB/SRA_2way", "weighted_f1")
# dataset_curve("BASELINE_RESULTS_SEP_DATASETS/Beetle/SRA_5way", "weighted_f1")
# dataset_curve("BASELINE_RESULTS_SEP_DATASETS/Beetle/SRA_2way", "weighted_f1")

dataset_curve("TOTAL_RESULTS_REDUCED/ASAP", "QWK")
# dataset_curve("TOTAL_RESULTS_REDUCED/SEB/SRA_5way", "weighted_f1")
# dataset_curve("TOTAL_RESULTS_REDUCED/SEB/SRA_2way", "weighted_f1")
# dataset_curve("TOTAL_RESULTS_REDUCED/Beetle/SRA_5way", "weighted_f1")
# dataset_curve("TOTAL_RESULTS_REDUCED/Beetle/SRA_2way", "weighted_f1")

# dataset_curve("EXP_RESULTS/SRA_5way", "weighted_f1")
# dataset_curve("EXP_RESULTS/SRA_2way", "weighted_f1")