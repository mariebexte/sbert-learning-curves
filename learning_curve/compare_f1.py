import os
import pandas as pd
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# For a prompt, plot F1 of the labels for the different methods
method_colors = {"SBERT_max": "pink", "SBERT": "red", "LR": "blue", "RF": "green", "SVM": "orange", "BERT": "purple"}        
method_styles = {"SBERT_max": "-", "SBERT": (0, (5, 1)), "LR": (0, (1, 1)), "RF": "-.", "SVM": "-", "BERT": (0, (3, 1, 1, 1))}

label_styles = {0: "-", 1: (0, (5, 1)), 2: (0, (1, 1)), 3: (0, (3, 1, 1, 1)),
                    "correct": "-", "incorrect": (0, (5, 1)), "contradictory": (0, (1, 1)), "partially_correct_incomplete": (0, (3, 1, 1, 1)), "non_domain": (0, (3, 1, 1, 1, 1, 1)), "irrelevant":(0, (3, 5, 1, 5))}
label_colors = {0: "red", 1: 'orange', 2: "blue", 3: "green",
                    "correct": "green", "incorrect": 'red', "contradictory": "blue", "partially_correct_incomplete": "orange", "non_domain": "purple", "irrelevant": "pink"}



def plot_method(labels, prompt_folder, method, pred_filename, color_by="label"):

    method_averages = pd.DataFrame()

    # For SBERT (method name can be SBERT_max, but result folder is named SBERT)
    if "_" in method:
        actual_method_name = method[:method.index("_")]
    else:
        actual_method_name = method

    # For each training size
    for training_size in os.listdir(os.path.join(prompt_folder, actual_method_name)):
        train_size_path = os.path.join(prompt_folder, actual_method_name, training_size)
        if os.path.isdir(train_size_path):

            train_size = int(training_size[training_size.rindex("_")+1:])
            # Collect results for this training size into a dataframe
            train_size_results = pd.DataFrame(columns=labels)

            # For each of the runs for this training size
            for sample_run in os.listdir(train_size_path):
                sample_run_path = os.path.join(train_size_path, sample_run)
                if os.path.isdir(sample_run_path):

                    # Calculate per-label F1 score and put into dataframe
                    df_pred = pd.read_csv(os.path.join(sample_run_path, pred_filename))
                    f1_results = f1_score(y_true=df_pred["y_true"], y_pred=df_pred["y_pred"], average=None, labels=labels)
                    df_f1_results = pd.DataFrame([f1_results], columns=labels)
                    train_size_results = pd.concat([train_size_results, df_f1_results])

            # Get average F1 per label for this training size, put into overall dataframe
            train_size_averages = train_size_results.mean()
            method_averages[train_size] = train_size_averages


    # Sort dataframe columns (ascending order of training size)
    col_list = method_averages.columns
    col_list = col_list.sort_values()
    method_averages=method_averages[col_list]

    style_dict = {}
    color_dict = {}

    # Plot, depending on color_by either use colors or styles to differentiate the labels
    for label, row in method_averages.iterrows():

        if color_by == 'label':
            plt.plot(row, linestyle=method_styles[method], color=label_colors[label]) 
            if not method in style_dict.keys():
                style_dict[method] = method_styles[method]
            if not label in color_dict.keys():
                color_dict[label] = label_colors[label]
        
        elif color_by == "method":
            plt.plot(row, linestyle=label_styles[label], color=method_colors[method]) 
            if not label in style_dict.keys():
                style_dict[label] = label_styles[label]
            if not method in color_dict.keys():
                color_dict[method] = method_colors[method]
        
        else:
            print("Unknown 'color_by': "+color_by)

    return style_dict, color_dict


# For a given prompt
# Calculate average F1 scores per label for the different training sizes

def plot_prompt(dataset_name, prompt_folder, strategy, test_file, color_by="label"):

    prompt_name = os.path.basename(prompt_folder)

    test_data = pd.read_csv(test_file)
    labels = list(test_data["label"].unique())
    support = test_data["label"].value_counts().to_dict()

    # For legend
    style_dict = {}
    color_dict = {}

    # For each method (but only consider LR SBERT BERT at this point to avoid overcrowding the graph)
    for method in os.listdir(prompt_folder):
        if os.path.isdir(os.path.join(prompt_folder, method)) and (method in ["LR", "SBERT", "BERT"]):

            style, color = plot_method(labels, prompt_folder, method, "predictions.csv", color_by=color_by)
            style_dict.update(style)
            color_dict.update(color)

            # For SBERT also consider predictions of max strategy
            if method == "SBERT":
                style, color = plot_method(labels, prompt_folder, method+"_max", "predictions_max.csv", color_by=color_by)
                style_dict.update(style)
                color_dict.update(color)


    # Add legend
    legend_entries = []
    for style in style_dict:
        if color_by=='method':
            legend_entries.append(Line2D([0], [0], color="black", linestyle=style_dict[style], lw=2, label=str(style)+" ("+str(support[style])+")"))
        else:
            legend_entries.append(Line2D([0], [0], color="black", linestyle=style_dict[style], lw=2, label=style))
    for color in color_dict:
        if color_by=='label':
            legend_entries.append(Line2D([0], [0], color=color_dict[color], lw=2, label=str(color)+" ("+str(support[color])+")"))
        else:
            legend_entries.append(Line2D([0], [0], color=color_dict[color], lw=2, label=str(color)))
    plt.legend(handles = legend_entries, loc='lower right')

    plt.title(dataset_name+": "+prompt_name+" ("+strategy+")", fontsize=12)
    plt.ylim([0, 1])
    plt.xlabel("# of training examples", fontsize=20)
    plt.ylabel("F1", fontsize=20)

    # Plot learning curve
    plt.grid()

    plt.rcParams['savefig.dpi'] = 300
    #plt.rc('legend', fontsize=4, handlelength=1)
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(prompt_folder, "f1_comparison_colorBy"+color_by+".png"))

    plt.clf()
    plt.cla()
    plt.close()



# For both sampling strategies
# Iterate over all prompts
# Each time looking up testing data
def plot_dataset(dataset_path, color_by='label'):

    dataset_name = os.path.basename(dataset_path)

    for sampling_strategy in ["balanced", "random"]:

        for prompt in os.listdir(os.path.join(dataset_path, sampling_strategy)):
            if os.path.isdir(os.path.join(dataset_path, sampling_strategy, prompt)):

                if "SRA" in dataset_name:   
                    annot_scheme = dataset_name
                    subset_name = dataset_path.split('/')[-2]
                    test_data = os.path.join("data", annot_scheme, subset_name, prompt, "test-unseen-answers.csv")
                else:
                    test_data = os.path.join("data", dataset_name, prompt, "test.csv")

                print(test_data)
                plot_prompt(dataset_name=dataset_name, prompt_folder=os.path.join(dataset_path, sampling_strategy, prompt), strategy=sampling_strategy, test_file=test_data, color_by=color_by)
                            

#for coloring_option in ["label", "method"]:
for coloring_option in ["method"]:

    plot_dataset("EXP_RESULTS_SEP_DATASETS/ASAP", color_by=coloring_option)
    plot_dataset("EXP_RESULTS_SEP_DATASETS/SEB/SRA_2way", color_by=coloring_option)
    plot_dataset("EXP_RESULTS_SEP_DATASETS/SEB/SRA_5way", color_by=coloring_option)
    plot_dataset("EXP_RESULTS_SEP_DATASETS/Beetle/SRA_2way", color_by=coloring_option)
    plot_dataset("EXP_RESULTS_SEP_DATASETS/Beetle/SRA_5way", color_by=coloring_option)
