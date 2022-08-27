import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curve(path, dataset_name, sampling_strategy, prompt_name, method_name, eval_measure, df_preds, ylim=[0,1]):

    train_sizes=df_preds.columns
    prompt_name = prompt_name.replace("/", "-")

    # In case of QWK, have to fisther transform before averaging
    if eval_measure.lower() == "qwk":
        # Arctanh == FISHER
        df_preds_fisher = np.arctanh(df_preds)
        test_scores_mean_fisher = np.nanmean(df_preds_fisher, axis=0)
        test_scores_std_fisher = np.nanstd(df_preds_fisher, axis=0)
        # Tanh == FISHERINV
        test_scores_mean = np.tanh(test_scores_mean_fisher)
        test_scores_std = np.tanh(test_scores_std_fisher)

    else:
        test_scores_mean = np.nanmean(df_preds, axis=0)
        test_scores_std = np.nanstd(df_preds, axis=0)

    test_scores_min = np.nanmin(df_preds, axis=0)
    test_scores_max = np.nanmax(df_preds, axis=0)

    plt.plot(
        train_sizes, test_scores_max, "o-", color="g", label="Best"
    )
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="b", label="Mean"
    )
    plt.plot(
        train_sizes, test_scores_min, "o-", color="r", label="Worst"
    )

    # Add standard deviation
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="blue",
    )

    plt.title(dataset_name+": "+prompt_name+" ("+method_name+", "+sampling_strategy+")", fontsize=12)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel("# of training examples", fontsize=20)
    plt.ylabel(eval_measure, fontsize=20)

    plt.grid()

    plt.rcParams['savefig.dpi'] = 300
    plt.rc('legend', fontsize=16, handlelength=1)
    
    plt.legend(loc="lower right")
    plt.xticks(train_sizes)
    #plt.xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.savefig(os.path.join(path, dataset_name+"_"+sampling_strategy+"_"+prompt_name +"_"+method_name+"_LC.png"))

    plt.clf()
    plt.cla()
    plt.close()