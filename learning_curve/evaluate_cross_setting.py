import os
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.transforms import TransformedBbox

# within_results_path = "FINAL_RESULTS/TOTAL_RESULTS_REDUCED/ASAP/random"
# cross_results_path = "FINAL_RESULTS/RESULTS_CROSS_FINAL/ASAP"
within_results_path = "results/TOTAL_RESULTS_REDUCED/ASAP/random"
cross_results_path = "results_cross/RESULTS_REDUCED/ASAP"

target_path = "_analysis/cross_matrixes"

mpl.rcParams.update({'font.size': 12})

# asap_prompt_order = [1, 2, 10, 5, 6, 3, 4, 7, 8, 9]
asap_prompt_order = ["1", "2", "10", "5", "6", "3", "4", "7", "8", "9"]


if not os.path.exists(target_path):
    os.makedirs(target_path)


# Assumes eval metric to be QWK
def get_average_difference(df_within, df_cross, training_sizes=None):

    if training_sizes is not None:

        df_within = df_within[training_sizes]
        df_cross = df_cross[training_sizes]


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
    # print("AVG DIFF", avg_difference_qwk)

    return avg_difference_qwk


def annotate_xranges(groups, ax=None):
    """
    Annotate a group of consecutive yticklabels with a group name.

    Arguments:
    ----------
    groups : dict
        Mapping from group label to an ordered list of group members.
    ax : matplotlib.axes object (default None)
        The axis instance to annotate.
    """
    if ax is None:
        ax = plt.gca()

    label2obj = {ticklabel.get_text() : ticklabel for ticklabel in ax.get_xticklabels()}

    for ii, (group, members) in enumerate(groups.items()):
        first = members[0]
        last = members[-1]

        bbox0 = _get_text_object_bbox(label2obj[first], ax)
        bbox1 = _get_text_object_bbox(label2obj[last], ax)

        set_xrange_label(group, bbox0.x0 + bbox0.width/2,
                         bbox1.x1 - bbox1.width/2,
                         min(bbox0.y0, bbox1.y0),
                         -0.6,
                         ax=ax)


def set_xrange_label(label, xmin, xmax, y, dy=-0.5, yskip=0.25, ax=None, *args, **kwargs):

    if not ax:
        ax = plt.gca()

    dx = xmax - xmin
    props = dict(connectionstyle='angle, angleA=180, angleB=90, rad=0',
                 arrowstyle='-',
                 shrinkA=2,
                 shrinkB=10,
                #  relpos=(0.5,-0.2),
                 lw=1)
    ax.annotate(label,
                ha='center',
                xy=(xmin, y + yskip),
                xytext=(xmin + dx/2, y + dy),
                annotation_clip=False,
                arrowprops=props,
                *args, **kwargs,
    )
    ax.annotate(label,
                ha='center',
                xy=(xmax, y + yskip),
                xytext=(xmin + dx/2, y + dy),
                annotation_clip=False,
                arrowprops=props,
                *args, **kwargs,
    )


def annotate_yranges(groups, ax=None):
    """
    Annotate a group of consecutive yticklabels with a group name.

    Arguments:
    ----------
    groups : dict
        Mapping from group label to an ordered list of group members.
    ax : matplotlib.axes object (default None)
        The axis instance to annotate.
    """
    if ax is None:
        ax = plt.gca()

    label2obj = {ticklabel.get_text() : ticklabel for ticklabel in ax.get_yticklabels()}

    for ii, (group, members) in enumerate(groups.items()):
        first = members[0]
        last = members[-1]

        bbox0 = _get_text_object_bbox(label2obj[first], ax)
        bbox1 = _get_text_object_bbox(label2obj[last], ax)

        set_yrange_label(group, bbox0.y0 + bbox0.height/2,
                         bbox1.y0 + bbox1.height/2,
                         min(bbox0.x0, bbox1.x0),
                         -2.5,
                         ax=ax)


def set_yrange_label(label, ymin, ymax, x, dx=-0.5, ax=None, *args, **kwargs):
    """
    Annotate a y-range.

    Arguments:
    ----------
    label : string
        The label.
    ymin, ymax : float, float
        The y-range in data coordinates.
    x : float
        The x position of the annotation arrow endpoints in data coordinates.
    dx : float (default -0.5)
        The offset from x at which the label is placed.
    ax : matplotlib.axes object (default None)
        The axis instance to annotate.
    """

    if not ax:
        ax = plt.gca()

    dy = ymax - ymin
    props = dict(connectionstyle='angle, angleA=90, angleB=180, rad=0',
                 arrowstyle='-',
                 shrinkA=10,
                 shrinkB=10,
                 lw=1)
    ax.annotate(label,
                xy=(x, ymin),
                xytext=(x + dx, ymin + dy/2),
                annotation_clip=False,
                arrowprops=props,
                *args, **kwargs,
    )
    ax.annotate(label,
                xy=(x, ymax),
                xytext=(x + dx, ymin + dy/2),
                annotation_clip=False,
                arrowprops=props,
                *args, **kwargs,
    )


def _get_text_object_bbox(text_obj, ax):
    # https://stackoverflow.com/a/35419796/2912349
    transform = ax.transData.inverted()
    # the figure needs to have been drawn once, otherwise there is no renderer?
    plt.ion(); plt.show(); plt.pause(0.001)
    bb = text_obj.get_window_extent(renderer = ax.get_figure().canvas.renderer)
    # handle canvas resizing
    return TransformedBbox(bb, transform)



def write_matrix(target_path, method, use_max=False, training_sizes=None, show_cbar=True):

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

                avg_difference = get_average_difference(df_within=results_within, df_cross=results_cross, training_sizes=training_sizes)
                target_results[base_prompt] = avg_difference


        df_target_results = pd.DataFrame.from_dict(target_results, orient='index', columns=[target_prompt]).T
        df_matrix = pd.concat([df_matrix, df_target_results])


    df_matrix = df_matrix[asap_prompt_order]
    # print("--- RESULT MATRIX: ---")
    # print(df_matrix)
    # print("---")

    suffix = ""
    if use_max:
        suffix = "_max"


    size_info = ""
    if training_sizes is not None:
        size_info = "-".join(training_sizes)
        target_path = os.path.join(target_path, size_info)
        size_info = "_"+size_info

    
    cbar_info = ""
    if show_cbar == False:
        cbar_info = "_no-cbar"

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    target_filename = os.path.join(target_path, method+suffix+size_info+cbar_info)

    df_matrix.to_csv(target_filename+".csv")

    cmap = sns.diverging_palette(20, 120, as_cmap=True)
    ax = heatmap = sns.heatmap(df_matrix, vmin=-.3, vmax=.3, center=0, annot=False, fmt='.2f', cmap=cmap, cbar=show_cbar, linewidth=.5, cbar_kws={'ticks': [-0.3, -0.2, -0.1, 0.0, +0.1, +0.2, +0.3], 'label': 'Change in QWK'}, annot_kws={"size":8})
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    # ax.set(xlabel="Base Prompt", ylabel="Target Prompt")
    # ax.set(ylabel="Target Prompt")
    plt.xlabel("Base Prompt", labelpad=30, fontweight='bold')
    plt.ylabel("Target Prompt", fontweight='bold')
    plt.yticks(rotation=0) 
    ax.hlines([3, 5], linewidth=1.5, color="black", *ax.get_xlim())
    ax.vlines([3, 5], linewidth=1.5, color="black", *ax.get_xlim())

    plt.subplots_adjust(top=0.8)
    # plt.subplots_adjust(left=0.3)
    groups = {
        'Science' : ("1", "2", "10"),
        'Bio' : ("5", "6"),
        'ELA' : ("3", "4", "7", "8", "9")
    }
    annotate_xranges(groups)
    # annotate_yranges(groups)

    clean_method_names = {'SBERT_max': "S-BERT"}
    method = clean_method_names.get(method+suffix, method+suffix)
    plt.title(method, y=-0.1)

    plt.rcParams['savefig.dpi'] = 500
    plt.savefig(target_filename+".pdf")

    plt.clf()
    plt.cla()
    plt.close()


for show_cbar in [True, False]:

    # write_matrix(target_path=target_path, method="SBERT", use_max=False, show_cbar=show_cbar)
    write_matrix(target_path=target_path, method="SBERT", use_max=True, show_cbar=show_cbar)
    # write_matrix(target_path=target_path, method="pretrained", use_max=False, show_cbar=show_cbar)
    # write_matrix(target_path=target_path, method="pretrained", use_max=True, show_cbar=show_cbar)
    # write_matrix(target_path=target_path, method="LR_easyadapt", use_max=False, show_cbar=show_cbar)
    # write_matrix(target_path=target_path, method="LR_merge", use_max=False, show_cbar=show_cbar)
    write_matrix(target_path=target_path, method="BERT", use_max=False, show_cbar=show_cbar)

    # for training_size in ['9', '14', '19', '24', '29', '34', '39', '44', '49', '54']:

    #     write_matrix(target_path=target_path, method="SBERT", use_max=False, training_sizes=[training_size], show_cbar=show_cbar)
    #     write_matrix(target_path=target_path, method="SBERT", use_max=True, training_sizes=[training_size], show_cbar=show_cbar)
    #     write_matrix(target_path=target_path, method="pretrained", use_max=False, training_sizes=[training_size], show_cbar=show_cbar)
    #     write_matrix(target_path=target_path, method="pretrained", use_max=True, training_sizes=[training_size], show_cbar=show_cbar)
    #     write_matrix(target_path=target_path, method="LR_easyadapt", use_max=False, training_sizes=[training_size], show_cbar=show_cbar)
    #     write_matrix(target_path=target_path, method="LR_merge", use_max=False, training_sizes=[training_size], show_cbar=show_cbar)
    #     write_matrix(target_path=target_path, method="BERT", use_max=False, training_sizes=[training_size], show_cbar=show_cbar)