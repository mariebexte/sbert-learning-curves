import os
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt

within_results_path = "FINAL_RESULTS/TOTAL_RESULTS_REDUCED/ASAP/random"
cross_results_path = "FINAL_RESULTS/RESULTS_CROSS_FINAL/ASAP"

target_path = "_analysis/cross_matrixes"

asap_prompt_order = [1, 2, 10, 5, 6, 3, 4, 7, 8, 9]


if not os.path.exists(target_path):
    os.makedirs(target_path)


# Assumes eval metric to be QWK
def get_average_difference(df_within, df_cross):

    # Arctanh == FISHER
    df_within_fisher = np.arctanh(df_within)
    df_cross_fisher = np.arctanh(df_cross)

    within_mean_fisher = np.nanmean(df_within_fisher, axis=0)
    cross_mean_fisher = np.nanmean(df_cross_fisher, axis=0)

    # print("within", within_mean_fisher)
    # print("cross", cross_mean_fisher)

    differences_fisher = cross_mean_fisher - within_mean_fisher
    avg_difference_fisher = np.nanmean(differences_fisher, axis=0)

    # print("differences", differences_fisher)
    # print("avg diff", avg_difference_fisher)

    # Tanh == FISHERINV
    avg_difference_qwk = np.tanh(avg_difference_fisher)

    return avg_difference_qwk



def write_matrix(target_path, method, use_max=False):

    method_within = method
    if "_" in method:
        method_within = method.split("_")[0]

    df_matrix = pd.DataFrame()

    for target_prompt in asap_prompt_order:

        target_results = {}

        for base_prompt in asap_prompt_order:

            # Only consider using a different prompt as base
            if not base_prompt == target_prompt:

                result_file_name = "QWK_lc_results.csv"
                if use_max:
                    result_file_name = "QWK_lc_results_max.csv"

                result_filename_within = os.path.join(within_results_path, str(target_prompt), method_within, result_file_name)
                result_filename_cross = os.path.join(cross_results_path, "base_"+str(base_prompt), "ASAP", "random", str(target_prompt), method, result_file_name)

                results_within = pd.read_csv(result_filename_within)
                results_cross = pd.read_csv(result_filename_cross)

                avg_difference = get_average_difference(df_within=results_within, df_cross=results_cross)
                target_results[base_prompt] = avg_difference


        df_target_results = pd.DataFrame.from_dict(target_results, orient='index', columns=[target_prompt]).T
        df_matrix = pd.concat([df_matrix, df_target_results])


    df_matrix = df_matrix[asap_prompt_order]
    print("--- RESULT MATRIX: ---")
    print(df_matrix)
    print("---")

    suffix = ""
    if use_max:
        suffix = "_max"
    target_filename = os.path.join(target_path, method+suffix)

    df_matrix.to_csv(target_filename+".csv")

    ax = heatmap = sns.heatmap(df_matrix, vmin=-.3, vmax=.3, center=0, cmap=sns.diverging_palette(20, 120, as_cmap=True), linewidth=.5, cbar_kws={'label': 'Change in QWK'})
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set(xlabel="Base Prompt", ylabel="Target Prompt")
    ax.hlines([3, 5], linewidth=2, color="black", *ax.get_xlim())
    ax.vlines([3, 5], linewidth=2, color="black", *ax.get_xlim())

    plt.rcParams['savefig.dpi'] = 300
    plt.savefig(target_filename+".png")

    plt.clf()
    plt.cla()
    plt.close()



write_matrix(target_path=target_path, method="SBERT", use_max=False)
write_matrix(target_path=target_path, method="SBERT", use_max=True)
write_matrix(target_path=target_path, method="pretrained", use_max=False)
write_matrix(target_path=target_path, method="pretrained", use_max=True)
write_matrix(target_path=target_path, method="LR_easyadapt", use_max=False)
write_matrix(target_path=target_path, method="LR_merge", use_max=False)
write_matrix(target_path=target_path, method="BERT", use_max=False)