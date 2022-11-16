import pstats
import pandas as pd
import sys
import os
import random
import logging
from datetime import datetime
from copy import deepcopy
import torch

from regex import F
from learning_curve.train_shallow import train_shallow
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, classification_report, f1_score
from learning_curve.plot_learning_curve import plot_learning_curve
from learning_curve.train_sbert import train_sbert
from learning_curve.train_bert import train_bert
from learning_curve.baselines import get_predictions
from learning_curve.pretrained_SBERT import get_predictions_pretrained
from sentence_transformers import SentenceTransformer

target_column = "label"
results_folder = "results"

def write_classification_statistics(filepath, y_true, y_pred, qwk):
    with open(filepath, 'w') as eval_stats:
        eval_stats.write(classification_report(y_true=y_true, y_pred=y_pred)+"\n\n")
        true_series = pd.Series(y_true, name='Actual')
        pred_series = pd.Series(y_pred, name='Predicted')
        eval_stats.write(str(pd.crosstab(true_series, pred_series))+"\n\n")
        eval_stats.write("QWK:\t"+str(qwk))


# dataset_name: Subdir of results folder where results will be written
# method: The algorithm for which to calculate the learning curve, can be 'LR', 'SVM', 'RF', 'BERT', 'SBERT'
# eval_measure: The measure to use for the learning curve: 'QWK' for ASAP, 'weighted_f1' for SRA
# sampling strategy: Either 'balanced' or 'random'. 'balanced' takes n per label for the n-th training size, 'random' takes n*num_labels.
# num_labels: Steps in training sizes are n*num_labels until maximum possible sample size
# num_samples: How many training subsets to sample per training size
# upsample_training: Only takes effect when sampling_strategy is 'balanced' and num_labels > number of actually present labels; Creating as-balanced-as-possible training sets in steps of num_labels
# num_sbert_pairs_per_example: If None, build as many pairs as possible, otherwise limit to the specified amount, but if possible select different pairs across epochs
def run(dataset_name, prompt_name, train_path, val_path, test_path, method, eval_measure, sampling_strategy, num_labels, max_size=None, upsample_training=False, num_samples=20, predetermined_train_sizes=None, bert_base="bert-base-uncased", sbert_base="all-MiniLM-L6-v2", num_sbert_pairs_per_example=None):

    target_path = os.path.join(results_folder, dataset_name, sampling_strategy, prompt_name, method)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Clear logger from previous runs
    log = logging.getLogger()
    handlers = log.handlers[:]
    for handler in handlers:
        log.removeHandler(handler)
        handler.close()

    logging.basicConfig(filename=os.path.join(target_path, datetime.now().strftime('logs_%H_%M_%d_%m_%Y.log')), filemode='w', level=logging.DEBUG)

    # Generate a fixed set of random seeds to choose the same training instances across multiple runs with different algorithms
    sampling_seed = 4334
    # For each different training size, draw one training data subsample using each of these seeds
    seeds = random.Random(sampling_seed).sample(range(1, 10000), num_samples)
    logging.info("Sampled "+str(num_samples)+" random seeds with seed "+str(sampling_seed)+": "+str(seeds))

    # Read data
    df_train = pd.read_csv(train_path)
    df_val = None
    if val_path is not None:
        df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    num_train = len(df_train)
    num_val = len(df_val)
    num_test = len(df_test)

    labels_in_target = set(df_train[target_column].unique())

    logging.info("Running "+method+" with "+sampling_strategy+" sampling strategy!")
    logging.info("Training: "+train_path+ " ("+str(num_train)+" instances in total)")
    logging.info("Validation: "+val_path+ " ("+str(num_val)+" instances in total)")
    logging.info("Testing: "+test_path+ " ("+str(num_test)+" instances in total)")

    # Determine training sizes based on available data and sampling strategy
    if sampling_strategy == 'balanced':

        # For balanced training: Ensure that every label is represented at least once (= as many labels in target column as num_labels)
        if len(labels_in_target) > num_labels:
            train_sizes=[]
            logging.info("Cannot run balanced learning curve: Training data has more than specified number of labels! Specified: "+str(num_labels)+" Present: "+str(len(labels_in_target)))
            print("More labels in data than specified num_labels!")

        elif (len(labels_in_target) < num_labels) and not upsample_training:
                train_sizes = []
                logging.info("Cannot run balanced learning curve: Training data has less than specified number of labels! Specified: "+str(num_labels)+" Present: "+str(len(labels_in_target))+" If you want to fill this discrepancy, set 'upsample_training' to True.")
                print("Not all labels are present in training data and 'upsample_training' is not set to True!")
                
        else:
            # As many labels present as specified OR less than that but wish to upsample
            # Determine least frequent label
            lowest_count = sys.maxsize
            for label in labels_in_target:
                label_count = len(df_train[df_train[target_column] == label])
                if label_count < lowest_count:
                    lowest_count = label_count
            # For balanced training, maximum training size is limited by least frequent label
            max_training_size = lowest_count*len(labels_in_target)

            # If predetermined training sizes were passed: Use these and check whether there is enough data to perform all of them
            if predetermined_train_sizes is not None:
                logging.info("A predetermined array of training sizes was passed, deciding whether we can calculate all of the requested sizes with the available data.")
                logging.info("Predetermined training sizes: "+str(predetermined_train_sizes))

                if max_size is not None:
                    logging.warn("Maximum training size (max_size) was set to '"+str(max_size)+"', but you also passed an array of predetermined training sizes. Ignoring max_size!") 

                # Remove all sizes that would exceed the available data
                while predetermined_train_sizes[-1] > max_training_size:
                    predetermined_train_sizes.pop()
                
                train_sizes = predetermined_train_sizes


            # If no predetermined training sizes were passed, determine them based on the available data
            else:
                logging.info("Determining training sizes based on available data.")

                # If a maximum size was specified and does not exceed the maximum possible size: Apply it
                if (max_size is not None) and (max_size < max_training_size):
                    max_training_size = max_size

                # Lowest: One per label, highest: max_training_size, in steps of num_labels
                train_sizes = list(range(num_labels, max_training_size + 1, num_labels))

            # Just to log that training samples will not be perfectly balanced
            if len(labels_in_target) < num_labels:
                logging.info("Number of labels in training data not equal to specified 'num_labels': Specified: "+str(num_labels)+" Present: "+str(len(labels_in_target))+". Because 'upsample_training' was set to 'True', training data will be upsampled to fit 'train_size' steps.")
                print("Less labels in training than number of specified labels, but 'upsample_training' is set to True, therefore upsampling!")

    elif sampling_strategy == 'random':

        # If predetermined training sizes were passed: Use these and check whether there is enough data to perform all of them
        if predetermined_train_sizes is not None:
            logging.info("A predetermined array of training sizes was passed, deciding whether we can calculate all of the requested sizes with the available data.")
            logging.info("Predetermined training sizes: "+str(predetermined_train_sizes))

            if max_size is not None:
                logging.warn("Maximum training size (max_size) was set to '"+str(max_size)+"', but you also passed an array of predetermined training sizes. Ignoring max_size!") 

            # Remove all sizes that would exceed the available data
            while predetermined_train_sizes[-1] > len(df_train):
                predetermined_train_sizes.pop()

            train_sizes = predetermined_train_sizes

        
        # If no predetermined training sizes were passed, determine them based on the available data
        else:
            logging.info("Determining training sizes based on available data.")

            # If no max_size was specified or max_size exceeds maximum possible size, run for as many as possible
            if (max_size is None) or (max_size > len(df_train)):
                max_size = len(df_train)
            train_sizes = list(range(num_labels, max_size + 1, num_labels))
    
    else:
        print("Unknown sampling strategy:", sampling_strategy, "! Please choose either 'balanced' or 'random'!")
        logging.error("Unknown sampling strategy: "+sampling_strategy+"! Please choose either 'balanced' or 'random'!")
        sys.exit(0)

    logging.info("Training sizes: "+str(train_sizes))
    print("Training sizes: "+str(train_sizes))
    
    # There are configurations where no run is possible
    if len(train_sizes) > 0:

        # Dataframe to keep track of performance of the different sample sizes, one row per run
        # As both training and validation instances have to be labeled, learning curve shoud be based on sum of training and validation data
        actual_train_sizes = [size + num_val for size in train_sizes]
        df_results = pd.DataFrame(columns=actual_train_sizes)

        # For SBERT: Keep second dataframe with results obtained using max strategy, default is avg
        if method in ["SBERT", "edit", "overlap", "pretrained", "cosine"]:
            df_results_max = pd.DataFrame(columns=actual_train_sizes)

        # For each of the different training data sizes:
        for train_size in train_sizes:

            logging.info("Starting training size "+str(train_size)+" (last one is "+str(train_sizes[-1])+")")
            start = datetime.now()

            # Collect results for the different runs for this training size
            train_size_results = []

            # For SBERT: Keep second list of results obtained using max strategy, default is avg
            if method in ["SBERT", "edit", "overlap", "pretrained", "cosine"]:
                train_size_results_max = []

            # Run num_sample runs for the current training size
            for sample in range(num_samples):

                # Prepare dir for result files
                run_path = os.path.join(target_path, "train_size_" + str(train_size+num_val), "sample_num_" + str(sample))
                if not os.path.exists(run_path):
                    os.makedirs(run_path)

                # 'random' samples train_size randomly
                if sampling_strategy == 'random':
                    df_train_subset = df_train.sample(train_size, random_state=seeds[sample])

                # 'balanced' samples train_size/num_label per label
                # Use the number of actually present labels in training (as opposed to num_labels), as this is also what was used to determine training sizes
                elif sampling_strategy == 'balanced':

                    # Sample train_size/num_labels_in_train per label
                    df_train_subset = pd.DataFrame(columns=df_train.columns)
                    for label in labels_in_target:
                        df_label = df_train[df_train[target_column] == label]
                        df_train_subset = pd.concat([df_train_subset, df_label.sample(int(train_size/len(labels_in_target)), random_state=seeds[sample])])

                    # If we already have the desired amount of samples: All is well
                    if len(df_train_subset) == train_size:
                        pass

                    # Should never happen, just a sanity check: We should never end up with more samples than the desired training size
                    elif len(df_train_subset) > train_size:
                        logging.info("When sampling, a subsample endet up larger than intended! Size: "+str(len(df_train_subset))+" , intended: "+str(train_size))
                        print("Training subset larger than expected!")
                        print(len(df_train_subset))
                        print(train_size)
                        sys.exit(0)

                    # Current subsample of training data is smaller than desired training size
                    else:
                        # Add more instances to reach desired training size
                        if upsample_training:
                            # Determine how many additional instances we need
                            num_diff_to_sample = train_size-len(df_train_subset)
                            num_labels_in_training=len(df_train["label"].unique())
                            # Should never happen, just a sanity check:
                            # If there were as many labels present as we need additional training data, we could have included one more of each in the initial sampling
                            if num_diff_to_sample > num_labels_in_training:
                                logging.info("The gap of samples still to sample ("+str(num_diff_to_sample)+") is larger than the number of labels present in the training data ("+str(num_labels_in_training)+")!")
                                print("More answers left to sample than number of labels in training data!")
                                sys.exit(0)
                            # Determine remaining traning data (without what has already been sampled)
                            df_train_copy = df_train.copy()
                            df_train_copy = df_train_copy.drop(df_train_subset.index)
                            # Sample 'balanced' from remaining data
                            num_sampled = 0
                            # Shuffle labels so that it is not always the first class that gets sampled first
                            rand_labels_in_target = list(deepcopy(labels_in_target))
                            random.Random(seeds[sample]).shuffle(rand_labels_in_target)
                            # As long as we still have less instances in training subset than we need, sample one from the next class
                            for label in rand_labels_in_target:
                                df_train_subset = pd.concat([df_train_subset, df_train_copy[df_train_copy[target_column] == label].sample(1, random_state=seeds[sample])])
                                num_sampled += 1
                                # If we have enough, stop
                                if num_sampled == num_diff_to_sample:
                                    break
                            
                            # Sanity check: Should never happen. At this point we should have exactly the desired amount of training instances
                            if not(len(df_train_subset) == train_size):
                                logging.info("After upsampling, the training subset does not have the desired size! Desired:"+train_size+", but is:"+len(df_train_subset)+"!")
                                print("After upsampling, the training subset does not have the desired size! Desired:"+train_size+", but is:"+len(df_train_subset)+"!")
                                sys.exit(0)

                        # Should never happen, just as a sanity check. Upon entering the current branch, upsample_training should always be True
                        else:
                            logging.info("When sampling, a subsample endet up smaller than intended, but 'upsample_training' was set to 'False'!")
                            print("Training subsample smaller than expected, and upsample_training is set to 'False'!")
                            sys.exit(0)

                else:
                    print("Unknown sampling strategy:", sampling_strategy, "! Please choose either 'balanced' or 'random'!")
                    logging.error("Unknown sampling strategy:", sampling_strategy, "! Please choose either 'balanced' or 'random'!")
                    sys.exit(0)

                # Tain model and get predictions
                if method in ["LR", "RF", "SVM"]:
                    y_pred = train_shallow(method=method, df_train=df_train_subset, df_val=df_val, df_test=df_test)

                elif method == "BERT":
                    y_pred = train_bert(run_path=run_path, df_train=df_train_subset, df_val=df_val, df_test=df_test, base_model=bert_base)

                elif method == "SBERT":
                    # Returns predictions obtained using max and avg (default) strategy
                    y_pred_max, y_pred = train_sbert(run_path=run_path, df_train=df_train_subset, df_val=df_val, df_test=df_test, base_model=sbert_base, num_pairs_per_example=num_sbert_pairs_per_example)

                elif method in ["edit", "overlap", "cosine"]:
                    y_pred_max, y_pred = get_predictions(method=method, df_train=df_train_subset, df_val=df_val, df_test=df_test)

                elif method == "pretrained":

                    pretrained_model = None
                    if method == "pretrained":
                        device = "cpu"
                        #device = "mps"
                        if torch.cuda.is_available():
                            device = "cuda"
                        pretrained_model = SentenceTransformer(sbert_base, device=device)

                    y_pred_max, y_pred = get_predictions_pretrained(df_train=df_train_subset, df_val=df_val, df_test=df_test, model=pretrained_model)

                else:
                    logging.error("Unknown prediction method: ", method, "! Please choose one of the following: 'LR', 'RF', 'SVM', 'BERT', 'SBERT'!")
                    print("Unknown prediction method: ", method, "! Please choose one of the following: 'LR', 'RF', 'SVM', 'BERT', 'SBERT'!")
                    sys.exit(0)

                # Calculate evaluation metrics
                y_true=list(df_test[target_column])

                qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
                weighted_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
                # For SBERT: Also calculate QWK and weighted F1 for predictions obtained using max strategy
                if method in ["SBERT", "edit", "overlap", "pretrained", "cosine"]:
                    qwk_max = cohen_kappa_score(y_true, y_pred_max, weights='quadratic')
                    weighted_f1_max = f1_score(y_true=y_true, y_pred=y_pred_max, average='weighted')

                # Write classification statistics to file
                write_classification_statistics(filepath=os.path.join(run_path, "results.txt"), y_true=y_true, y_pred=y_pred, qwk=qwk)
                # For SBERT: Also write classification statistics for predictions obtained using max strategy
                if method in ["SBERT", "edit", "overlap", "pretrained", "cosine"]:
                    write_classification_statistics(filepath=os.path.join(run_path, "results_max.txt"), y_true=y_true, y_pred=y_pred_max, qwk=qwk_max)

                # Write training instances to file
                df_train_subset.to_csv(os.path.join(run_path, "train.csv"), index=None)

                # Write predictions to file
                df_predictions = pd.DataFrame({'id': list(df_test["id"]), 'y_true': y_true, 'y_pred': y_pred})
                df_predictions.to_csv(os.path.join(run_path, "predictions.csv"), index=None)
                # For SBERT: Also write predictions obtained using max strategy
                if method in ["SBERT", "edit", "overlap", "pretrained", "cosine"]:
                    df_predictions_max = pd.DataFrame({'id': list(df_test["id"]), 'y_true': y_true, 'y_pred': y_pred_max})
                    df_predictions_max.to_csv(os.path.join(run_path, "predictions_max.csv"), index=None)

                # Append result for desired metric to learning curve statistics
                if eval_measure == 'QWK':
                    train_size_results.append(qwk)
                    # For SBERT: Also append to training size results for predictions obtained using the max strategy
                    if method in ["SBERT", "edit", "overlap", "pretrained", "cosine"]:
                        train_size_results_max.append(qwk_max)

                elif eval_measure == 'weighted_f1':
                    train_size_results.append(weighted_f1)
                    # For SBERT: Also append to training size results for predictions obtained using the max strategy
                    if method in ["SBERT", "edit", "overlap", "pretrained", "cosine"]:
                        train_size_results_max.append(weighted_f1_max)

                else:
                    print("Unknown evaluation measure:", eval_measure, "! Please choose either 'QWK' or 'weighted_f1'!")
                    logging.error("Unknown evaluation measure:", eval_measure, "! Please choose either 'QWK' or 'weighted_f1'!")
                    sys.exit(0)


            train_size_duration = datetime.now() - start
            logging.info(str(num_samples)+" runs of training size "+str(train_size)+" took "+str(train_size_duration)+" to run!")

            # Add results to all runs of this training size as a column of overall results dataframe
            df_results[train_size+num_val] = train_size_results
            # Save, to avoid results in case run is terminated prematurely
            df_results.to_csv(os.path.join(target_path, eval_measure + "_lc_results.csv"), index = None)
            # For SBERT: Also append to and save results dataframe for predictions obtained using the max strategy
            if method in ["SBERT", "edit", "overlap", "pretrained", "cosine"]:
                df_results_max[train_size+num_val] = train_size_results_max
                df_results_max.to_csv(os.path.join(target_path, eval_measure + "_lc_results_max.csv"), index = None)

        # Once all runs for all training sizes are finished, plot learning curve
        plot_learning_curve(path=target_path, dataset_name=dataset_name, sampling_strategy=sampling_strategy, prompt_name=prompt_name, method_name=method, eval_measure=eval_measure, df_preds=df_results)
        # For SBERT also plot curve of results obtained using the max strategy
        if method in ["SBERT", "edit", "overlap", "pretrained", "cosine"]:
            plot_learning_curve(path=target_path, dataset_name=dataset_name, sampling_strategy=sampling_strategy, prompt_name=prompt_name, method_name=method+"-max", eval_measure=eval_measure, df_preds=df_results_max)
