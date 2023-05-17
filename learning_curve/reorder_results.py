import os
from distutils.dir_util import copy_tree

# Change result structure: Group into Beetle and SEB datasets
def reorder(dir_to_reorder, target_location):

    dataset = os.path.basename(dir_to_reorder)

    # Go from SRA_5way/random/Beetle to Beetle/SRA_5way/random
    for strategy in os.listdir(dir_to_reorder):
        if os.path.isdir(os.path.join(dir_to_reorder, strategy)):

            for subset in os.listdir(os.path.join(dir_to_reorder, strategy)):
                if os.path.isdir(os.path.join(dir_to_reorder, strategy, subset)):

                    source_path = os.path.join(dir_to_reorder, strategy, subset)
                    target_path = os.path.join(target_location, subset, dataset, strategy)

                    # Copy all contents of source_path to target_path
                    copy_tree(source_path, target_path)


reorder("results/SRA_5way", "results/RESULTS_SEP_DATASETS")
reorder("results/SRA_2way", "results/RESULTS_SEP_DATASETS")