import os
from distutils.dir_util import copy_tree
from shutil import copyfile, copy
import os

# Grab just the result tables from the learning curve runs and copy to a new location
def collect(dir_to_collect_from, target_location):

    dataset = os.path.basename(dir_to_collect_from)

    for dataset in ["ASAP"]:
        if os.path.isdir(os.path.join(dir_to_collect_from, dataset)):
    
            for strategy in ["balanced", "random"]:
                if os.path.isdir(os.path.join(dir_to_collect_from, dataset, strategy)):

                    for prompt in os.listdir(os.path.join(dir_to_collect_from, dataset, strategy)):
                        if os.path.isdir(os.path.join(dir_to_collect_from, dataset, strategy, prompt)):

                            for method in os.listdir(os.path.join(dir_to_collect_from, dataset, strategy, prompt)):
                                if os.path.isdir(os.path.join(dir_to_collect_from, dataset, strategy, prompt, method)):

                                    source_path = os.path.join(dir_to_collect_from, dataset, strategy, prompt, method, "QWK_lc_results.csv")
                                    target_path = os.path.join(target_location, dataset, strategy, prompt, method, "QWK_lc_results.csv")

                                    source_path_max = os.path.join(dir_to_collect_from, dataset, strategy, prompt, method, "QWK_lc_results_max.csv")
                                    target_path_max = os.path.join(target_location, dataset, strategy, prompt, method, "QWK_lc_results_max.csv")

                                    if not os.path.exists(os.path.join(target_location, dataset, strategy, prompt, method)):
                                        os.makedirs(os.path.join(target_location, dataset, strategy, prompt, method))

                                    try:
                                        copyfile(source_path, target_path)
                                    except:
                                        pass

                                    try:
                                        copy(source_path_max, target_path_max)
                                    except:
                                        pass



    for dataset in ["SEB", "BEETLE"]:
        if os.path.isdir(os.path.join(dir_to_collect_from, dataset)):

            for subset in ["SRA_2way", "SRA_5way"]:
                if os.path.isdir(os.path.join(dir_to_collect_from, dataset, subset)):

                    for strategy in ["balanced", "random"]:
                        if os.path.isdir(os.path.join(dir_to_collect_from, dataset, subset, strategy)):

                            for prompt in os.listdir(os.path.join(dir_to_collect_from, dataset, subset, strategy)):
                                if os.path.isdir(os.path.join(dir_to_collect_from, dataset, subset, strategy, prompt)):

                                    for method in os.listdir(os.path.join(dir_to_collect_from, dataset, subset, strategy, prompt)):
                                        if os.path.isdir(os.path.join(dir_to_collect_from, dataset, subset, strategy, prompt, method)):

                                            source_path = os.path.join(dir_to_collect_from, dataset, subset, strategy, prompt, method, "weighted_f1_lc_results.csv")
                                            target_path = os.path.join(target_location, dataset, subset, strategy, prompt, method, "weighted_f1_lc_results.csv")

                                            source_path_max = os.path.join(dir_to_collect_from, dataset, subset, strategy, prompt, method, "weighted_f1_lc_results_max.csv")
                                            target_path_max = os.path.join(target_location, dataset, subset, strategy, prompt, method, "weighted_f1_lc_results_max.csv")

                                            if not os.path.exists(os.path.join(target_location, dataset, subset, strategy, prompt, method)):
                                                os.makedirs(os.path.join(target_location, dataset, subset, strategy, prompt, method))

                                            try:
                                                copyfile(source_path, target_path)
                                            except:
                                                pass

                                            try:
                                                copyfile(source_path_max, target_path_max)
                                            except:
                                                pass



collect("BASELINE_RESULTS_SEP_DATASETS", "TOTAL_RESULTS_REDUCED")
collect("EXP_RESULTS_SEP_DATASETS", "TOTAL_RESULTS_REDUCED")