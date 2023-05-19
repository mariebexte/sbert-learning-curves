import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pylab import plot, legend
import os
import pandas as pd
import numpy as np

mpl.rcParams.update({'font.size': 19, 'legend.loc': 'upper center'})
analysis_folder = "_analysis"


def plot_boxes(sd_lists, labels, target_file, title, methods):

    clean_names = {'SBERT_max': 'SBERT'}

    colors={"BERT": "#0073DF", "SBERT_max": "#C70000", 'LR': '#00E7B5', "edit_max": '#EB8500', "pretrained_max": '#C70000'}
    colors={"BERT": "#0073DF", "SBERT_max": "#5bc99c", 'LR': '#00E7B5', "edit_max": '#EB8500', "pretrained_max": '#C70000'}
    font_size=24

    meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=5)
    medianprops = dict(linestyle='-', color='white')
    fig, ax = plt.subplots()
    x = np.array(labels)

    if len(methods) == 2:
        offsets = [-1.1, 1.1]
        width = 1.5
    elif len(methods) == 3:
        offsets = [-1.2, 0, 1.2]
        width = 1.0

    bplots = []
    for i in range(len(methods)):
        method_data = sd_lists[methods[i]]
        bplot = ax.boxplot(method_data.values(),0,'',positions=x+offsets[i],widths=width, showmeans=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
        bplots.append(bplot)

    # data_bert = sd_lists["BERT"]
    # data_sbert = sd_lists["SBERT"]
    # # plt.figure()
    # bplotbert = ax.boxplot(data_bert.values(),0,'',positions=x-1.1,widths=1.5, showmeans=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
    # bplotsbert = ax.boxplot(data_sbert.values(),0,'',positions=x+1.1,widths=1.5, showmeans=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)

    # fill with colors
    # for (bplot, color) in [(bplotbert, colors['BERT']), (bplotsbert, colors['SBERT'])]:
    #     for patch in bplot['boxes']:
    #         patch.set_facecolor(color)

    plotcolors = [colors[method] for method in methods]
    for bplot, color in zip(bplots, plotcolors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)
    
    # custom_legend_entries = [Line2D([0], [1], color=colors['BERT'], marker='|', linestyle='None',
    #                       markersize=12, markeredgewidth=10),
    #             Line2D([0], [0], color=colors['SBERT'], marker='|', linestyle='None',
    #                       markersize=12, markeredgewidth=10)]

    custom_legend_entries = []
    for method in methods:
        custom_legend_entries.append(Line2D([0], [1], color=colors[method], marker='|', linestyle='None', markersize=12, markeredgewidth=10))

    clean_methods = [clean_names.get(method, method) for method in methods]
    ax.legend(custom_legend_entries, clean_methods, ncol=len(methods))

    ax.set_xlabel('# of Training Examples')
    ax.set_ylabel('QWK SD')

    plt.xticks(x)
    string_labels = [str(size) for size in labels]
    ax.set_xticklabels(string_labels)

    plt.ylim([0, 0.38])
    plt.yticks([0, 0.1, 0.2, 0.3])

    plt.title(title)
    plt.tight_layout()
    plt.savefig(target_file)
    # plt.show()




# # Dict: Training size -> Method
# # Dict: Method -> List of SDs for the different prompts
# sd_dict = {}

# for method in ["BERT", "SBERT"]:

#     method_results = pd.read_csv(os.path.join(base_path+method+'_prompt-comparison_SD.csv'))
    
#     for training_size in method_results.columns:

#         try: 
#             training_size = int(training_size)
#             size_dict = sd_dict.get(training_size, {})
#             size_dict[method] = list(method_results[str(training_size)])
#             sd_dict[training_size] = size_dict

#         except ValueError:
#             print("skipping column", training_size, "because it does not look like a training size")


# methods = ['pretrained_max', 'edit_max']
# methods = ["LR", "BERT"]
# methods = ["SBERT_max", 'pretrained_max', 'edit_max']
# methods = ["LR", "BERT", "SBERT_max", 'pretrained_max', 'edit_max']
methods = ["BERT", "SBERT_max"]

### ASAP balanced
# base_path = "FINAL_RESULTS/TOTAL_RESULTS_REDUCED/ASAP/balanced/ASAP_balanced_"
base_path = "results/RESULTS_REDUCED/ASAP/balanced/ASAP_balanced_"
overall_dict = {}

for method in methods:

    method_dict = {}
    method_results = pd.read_csv(os.path.join(base_path+method+'_prompt-comparison_SD.csv'))
    
    for training_size in method_results.columns:

        try: 
            training_size = int(training_size)
            method_dict[training_size] = list(method_results[str(training_size)])

        except ValueError:
            print("skipping column", training_size, "because it does not look like a training size")
    
    overall_dict[method] = method_dict

plot_boxes(overall_dict, list(method_dict.keys()), os.path.join(analysis_folder, "SD_ASAP_balanced.pdf"), 'Balanced Sampling', methods)



### ASAP random
# base_path = "FINAL_RESULTS/TOTAL_RESULTS_REDUCED/ASAP/random/ASAP_random_"
base_path = "results/RESULTS_REDUCED/ASAP/random/ASAP_random_"
overall_dict = {}

for method in methods:

    method_dict = {}
    method_results = pd.read_csv(os.path.join(base_path+method+'_prompt-comparison_SD.csv'))
    
    for training_size in method_results.columns:

        try: 
            training_size = int(training_size)
            method_dict[training_size] = list(method_results[str(training_size)])

        except ValueError:
            print("skipping column", training_size, "because it does not look like a training size")
    
    overall_dict[method] = method_dict

plot_boxes(overall_dict, list(method_dict.keys()), os.path.join(analysis_folder, "SD_ASAP_random.pdf"), 'Random Sampling', methods)






