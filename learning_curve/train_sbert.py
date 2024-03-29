import os
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.evaluation import SimilarityFunction
import pandas as pd
from torch.utils.data import DataLoader
import torch
import shutil
import sys
from scipy import spatial
import shutil
import logging


def eval_sbert(run_path, df_test, df_ref, id_column, answer_column, target_column):

    max_predictions = []
    avg_predictions = []

    # Later used to create dataframe with classification results
    predictions = {}
    predictions_index = 0

    # Cross every test embedding with every train embedding
    for idx, test_answer in df_test.iterrows():

        # Copy reference answer dataframe
        copy_eval = df_ref[[id_column, answer_column, target_column, "embedding"]].copy()
        # Put reference answers as 'answers 2'
        copy_eval.columns = ["id2", "text2", "score2", "embedding2"]
        # Put current testing answer as 'answer 1': Copy it num_ref_answers times into the reference dataframe to compare it to all of the reference answers
        copy_eval["id1"] = [test_answer[id_column]]*len(copy_eval)
        copy_eval["text1"] = [test_answer[answer_column]]*len(copy_eval)
        copy_eval["score1"] = [test_answer[target_column]]*len(copy_eval)
        copy_eval["embedding1"] = [test_answer["embedding"]]*len(copy_eval)
        emb1 = list(copy_eval["embedding1"])
        emb2 = list(copy_eval["embedding2"])
        copy_eval["cos_sim"] = [1 - spatial.distance.cosine(emb1[i], emb2[i]) for i in range(len(copy_eval))]

        # Determine prediction: MAX
        max_row = copy_eval.iloc[[copy_eval["cos_sim"].argmax()]]
        max_sim = max_row.iloc[0]["cos_sim"]
        max_pred = max_row.iloc[0]["score2"]
        max_sim_id = max_row.iloc[0]["id1"]
        max_sim_answer = max_row.iloc[0]["text1"]

        # Determine prediction: AVG
        label_avgs = {}
        for label in set(copy_eval["score2"]):
            label_subset = copy_eval[copy_eval["score2"] == label]
            label_avgs[label] = label_subset["cos_sim"].mean()
        avg_pred = max(label_avgs, key=label_avgs.get)
        avg_sim = max(label_avgs.values())

        max_predictions.append(max_pred)
        avg_predictions.append(avg_pred)

        predictions[predictions_index] = {"id": test_answer[id_column], "pred_avg": avg_pred, "sim_score_avg": avg_sim,"pred_max": max_pred, "sim_score_max": max_sim, "most_similar_answer_id": max_sim_id, "most_similar_answer_text": max_sim_answer}
        predictions_index += 1

    copy_test = df_test.copy()
    df_predictions = pd.DataFrame.from_dict(predictions, orient='index')
    df_predictions = pd.merge(copy_test, df_predictions, left_on=id_column, right_on="id")
    df_predictions.to_csv(os.path.join(run_path, "predictions_sim.csv"), index=None)

    return max_predictions, avg_predictions


# For larger amounts of training data: Do not create all possible pairs, but limit to a fixed number per epoch (if possible, have different pairs across different epochs)
def train_sbert(run_path, df_train, df_test, df_val, answer_column="text", target_column="label", id_column="id", base_model="all-MiniLM-L6-v2", batch_size=8, num_epochs=20, num_pairs_per_example=None, do_warmup=False, save_model=False):

    if num_pairs_per_example is not None:
        num_samples = len(df_train) * num_pairs_per_example
        num_batches_per_round = int(num_samples/batch_size)
        logging.info("LIMITING SBERT TRAINING PAIRS: "+str(num_pairs_per_example)+" pairs per sample!")

    device = "cpu"
    #device = "mps"
    if torch.cuda.is_available():
        device = "cuda"

    model = SentenceTransformer(base_model, device=device)

    # Where to store finetuned model: In LC this is just temporary, will be deleted at the end of the run
    model_path = os.path.join(run_path, "finetuned_model")

    # Define list of training pairs: Create as many as possible
    train_examples = []
    for _, example_1 in df_train.iterrows():
        for _, example_2 in df_train.iterrows():

            if not example_1[id_column] == example_2[id_column]:

                label = 0
                if example_1[target_column] == example_2[target_column]:
                    label = 1

                train_examples.append(InputExample(texts=[example_1[answer_column], example_2[answer_column]], label=label*1.0))

    # Define validation pairs: Create as many as possible
    val_example_dict = {}
    val_example_index = 0
    for _, example_1 in df_val.iterrows():
        for _, example_2 in df_train.iterrows():

            if not example_1[id_column] == example_2[id_column]:

                label = 0
                if example_1[target_column] == example_2[target_column]:
                    label = 1

                val_example_dict[val_example_index] = {"text_1": example_1[answer_column], "text_2": example_2[answer_column], "sim_label": label}
                val_example_index += 1

    val_examples = pd.DataFrame.from_dict(val_example_dict, "index")

    # Define train dataset, dataloader, train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.OnlineContrastiveLoss(model)

    # Define evaluator
    evaluator = evaluation.EmbeddingSimilarityEvaluator(val_examples["text_1"].tolist(), val_examples["text_2"].tolist(), val_examples["sim_label"].tolist())

    num_warm_steps = 0
    if do_warmup == True:
        steps_per_epoch = len(train_examples)/batch_size
        total_num_steps = steps_per_epoch * num_epochs
        num_warm_steps = round(0.1*total_num_steps)

    # Tune the model
    if num_pairs_per_example is not None:
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=num_warm_steps, evaluator=evaluator, output_path=model_path, save_best_model=True, show_progress_bar=True, steps_per_epoch=num_batches_per_round)
    else:
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=num_warm_steps, evaluator=evaluator, output_path=model_path, save_best_model=True, show_progress_bar=True)

    logging.info("SBERT number of epochs: "+str(num_epochs))
    logging.info("SBERT batch size: "+str(batch_size))
    logging.info("SBERT warmup steps: "+str(num_warm_steps))
    logging.info("SBERT evaluator: "+str(evaluator.__class__)+" Batch size: "+str(evaluator.batch_size)+" Main similarity:"+str(evaluator.main_similarity))
    logging.info("SBERT loss: "+str(train_loss.__class__))

    # Evaluate best model: Can only do this if training was sucessful, otherwise keep pretrained
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        model = SentenceTransformer(model_path)
    else:
        model = SentenceTransformer(base_model)

    # Eval testing data: Get sentence embeddings for all testing and reference answers
    df_test['embedding'] = df_test[answer_column].apply(model.encode)

    df_ref = pd.concat([df_val, df_train])
    df_ref['embedding'] = df_ref[answer_column].apply(model.encode)

    # Copy training statistic into run folder
    if os.path.exists(model_path):
        shutil.copyfile(os.path.join(model_path, "eval", "similarity_evaluation_results.csv"), os.path.join(run_path, "eval_training.csv"))

    # Delete model to save space
    if os.path.exists(model_path) and save_model==False:
        shutil.rmtree(model_path)

    return eval_sbert(run_path=run_path, df_test=df_test, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)