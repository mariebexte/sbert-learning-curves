from cmath import log
from transformers import Trainer, TrainingArguments, BertTokenizerFast, BertForSequenceClassification, TrainerCallback
import torch
import logging
import os
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, cohen_kappa_score
import pandas as pd
import shutil
from copy import deepcopy


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


class GetTestPredictionsCallback(TrainerCallback):

    def __init__(self, *args, dict_test_preds, save_path, trainer, test_data, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_test_preds = dict_test_preds
        self.save_path = save_path
        self.trainer = trainer
        self.test_data=test_data
        self.df_test_stats = pd.DataFrame()

    def on_log(self, args, state, control, logs=None, **kwargs):

        pred = self.trainer.predict(self.test_data)
        predictions = pred.predictions.argmax(axis=1)
        self.dict_test_preds[logs['epoch']] = predictions

        self.df_test_stats = pd.concat([self.df_test_stats, pd.DataFrame(pred.metrics, index=[int(logs['epoch'])])])
        # print(self.df_test_stats)

    def on_train_end(self, args, state, control, **kwargs):
        self.df_test_stats.to_csv(self.save_path, index_label='epoch')


# To log loss of training and evaluation to file
class WriteCsvCallback(TrainerCallback):

    def __init__(self, *args, csv_train, csv_eval, dict_val_loss, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_train_path = csv_train
        self.csv_eval_path = csv_eval
        self.df_eval = pd.DataFrame()
        self.df_train_eval = pd.DataFrame()
        self.dict_val_loss = dict_val_loss

    def on_log(self, args, state, control, logs=None, **kwargs):

        df_log = pd.DataFrame([logs])

        # Has info about performance on training data
        if "loss" in logs:
            self.df_train_eval = pd.concat([self.df_train_eval, df_log])
        
        # Has info about performance on validation data
        else:
            best_model = state.best_model_checkpoint
            df_log["best_model_checkpoint"] = best_model
            self.df_eval = pd.concat([self.df_eval, df_log])
            if 'eval_loss' in logs:
                self.dict_val_loss[int(logs['epoch'])] = logs['eval_loss']

    def on_train_end(self, args, state, control, **kwargs):
        self.df_eval.to_csv(self.csv_eval_path)
        self.df_train_eval.to_csv(self.csv_train_path)


# Which metrics to compute on evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    qwk = cohen_kappa_score(labels, preds, weights='quadratic') 
    return {
      'acc': acc,
      'weighted f1': f1,
      'qwk': qwk,
    }


def train_bert(run_path, df_train, df_val, df_test, answer_column="text", target_column="label", base_model="bert-base-uncased", num_epochs=20, batch_size=8, do_warmup=False, save_model=False):

    max_length = 512

    device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'

    # If all training instances have the same label: Return this label as prediction for all testing instances
    if len(df_train[target_column].unique()) == 1:
        target_label = list(df_train[target_column].unique())
        logging.warn("All training instances have the same label '"+str(target_label[0])+"'. Predicting this label for all testing instances!")
        print("All training instances have the same label '"+str(target_label[0])+"'. Predicting this label for all testing instances!")
        return target_label*len(df_test)

    # Model evaluation throws error if val/test data contains more labels than train
    labels_in_training = df_train[target_column].unique().tolist()
    labels_in_validation = df_val[target_column].unique().tolist()
    labels_in_test = df_test[target_column].unique().tolist()

    label_set = set(labels_in_training + labels_in_validation + labels_in_test)

    # If the labels are not integers: Map them to integers
    labels_are_string = False
    if(df_train[target_column].dtype == object):

        labels_are_string = True

        int_to_label = {}
        label_index = 0
        # Assign each label its designated integer
        for label in label_set:
            int_to_label[label_index] = label
            label_index += 1
                    
        # Create reversed map from label to integer
        label_to_int = {v: k for k, v in int_to_label.items()}
        logging.info("Mapping labels to integers: "+str(label_to_int))

        # Replace labels with their respective integers, use copies to avoid changing the original df
        df_train = df_train.copy()
        df_val = df_val.copy()
        df_test = df_test.copy()

        df_train[target_column] = [label_to_int[label] for label in df_train[target_column]]
        df_val[target_column] = [label_to_int[label] for label in df_val[target_column]]
        df_test[target_column] = [label_to_int[label] for label in df_test[target_column]]

    # Grab X,y for train, val and test
    train_texts = list(df_train.loc[:, answer_column])
    train_labels = list(df_train.loc[:, target_column])
    valid_texts = list(df_val.loc[:, answer_column])
    valid_labels = list(df_val.loc[:, target_column])
    test_texts = list(df_test.loc[:, answer_column])
    test_labels = list(df_test.loc[:, target_column])

    tokenizer = BertTokenizerFast.from_pretrained(base_model, do_lower_case=True)

    # Tokenize the dataset, truncate if longer than max_length, pad with 0's when less than `max_length`
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    # Convert tokenized data into a torch Dataset
    train_dataset = Dataset(train_encodings, train_labels)
    valid_dataset = Dataset(valid_encodings, valid_labels)
    test_dataset = Dataset(test_encodings, test_labels)

    num_warm_steps = 0
    if do_warmup == True:
        steps_per_epoch = len(df_train)/batch_size
        total_num_steps = steps_per_epoch * num_epochs
        num_warm_steps = round(0.1*total_num_steps)

    # Load model and pass to device
    model = BertForSequenceClassification.from_pretrained(base_model, num_labels=len(label_set), ignore_mismatched_sizes=True).to(device)
    model.train()

    training_args = TrainingArguments(
        output_dir=os.path.join(run_path, 'checkpoints'),   # Output directory to save model, will be deleted after evaluation
        num_train_epochs=num_epochs,             
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,  
        warmup_steps=num_warm_steps,
        # load_best_model_at_end=True,                        # Load the best model when finished training (default metric is loss)
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        # save_strategy="epoch",
        # save_total_limit=5,
        #weight_decay=0.01,
        #learning_rate=2e-5,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,                    # Callback that computes metrics of interest
        # callbacks=[]  # Callback to log loss during training
    )

    dict_val_loss = {}
    dict_test_preds = {}

    trainer.add_callback(WriteCsvCallback(csv_train=os.path.join(run_path, "train_stats.csv"), csv_eval=os.path.join(run_path, "eval_stats.csv"), dict_val_loss=dict_val_loss))
    trainer.add_callback(GetTestPredictionsCallback(dict_test_preds=dict_test_preds, save_path=os.path.join(run_path, "test_stats.csv"), trainer=trainer, test_data=test_dataset))
    trainer.train()

    # Determine epoch with lowest validation loss
    best_epoch = min(dict_val_loss, key=dict_val_loss.get)

    # For this epoch, return test predictions
    predictions = dict_test_preds[best_epoch]

    if labels_are_string:
        predictions = [int_to_label[pred] for pred in predictions]

    # if save_model == True:
    #     trainer.save_model(os.path.join(run_path, "best_model"))
    #     tokenizer.save_pretrained(os.path.join(run_path, "best_model"))

    # Delete model checkpoints to save space
    # if os.path.exists(os.path.join(run_path, "checkpoints")):
    #     shutil.rmtree(os.path.join(run_path, "checkpoints"), ignore_errors=True)

    return predictions