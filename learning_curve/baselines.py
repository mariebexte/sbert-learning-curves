from Levenshtein import distance
import pandas as pd
import sys
import os
from nltk.tokenize import word_tokenize
from collections import Counter
import math
from scipy import spatial

# Methods to calculate baseline similarity measures (not requiring any training and instead using training data as reference answers)
def get_overlap_score(answer_1, answer_2):

    a = word_tokenize(answer_1)
    b = word_tokenize(answer_2)

    # Length of each and intersection
    try:
        len_a = len(a)
        len_b = len(b)
        inter = 0
        for tok in (set(a) & set(b)):
            inter += min(a.count(tok), b.count(tok))
        overlap_1 = inter / len_a
        overlap_2 = inter / len_b
    except:  # at least one of the input is NaN
        overlap_1 = 0
        overlap_2 = 0
    
    return (overlap_1 + overlap_2) / 2


def get_cosine(vec1, vec2):

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def get_overlap_similarity(answer_1, answer_2):

    a = word_tokenize(answer_1)
    b = word_tokenize(answer_2)
    a_counts = Counter(a)
    b_counts = Counter(b)

    return get_cosine(a_counts, b_counts)
    

def get_predictions(run_path, method, df_train, df_val, df_test, id_column="id", answer_column="text", target_column="label"):

    max_predictions = []
    avg_predictions = []

    df_ref = pd.concat([df_train, df_val])

    # Later used to create dataframe with classification results
    predictions = {}
    predictions_index = 0

    # Compare every test instance with every train instance
    for idx, test_answer in df_test.iterrows():

        # Copy reference answer dataframe
        copy_eval = df_ref[[id_column, answer_column, target_column]].copy()
        # Put reference answers as 'answers 2'
        copy_eval.columns = ["id2", "text2", "score2"]
        # Put current testing answer as 'answer 1': Copy it num_ref_answers times into the reference dataframe to compare it to all of the reference answers
        copy_eval["id1"] = [test_answer[id_column]]*len(copy_eval)
        copy_eval["text1"] = [test_answer[answer_column]]*len(copy_eval)
        copy_eval["score1"] = [test_answer[target_column]]*len(copy_eval)
        # TODO: Plug in method
        ref_answers = list(copy_eval["text2"])
        test_answer_copies = list(copy_eval["text1"])
        if method == "edit":
            copy_eval["compare"] = [-1*distance(ref_answers[i], test_answer_copies[i]) for i in range(len(copy_eval))]
        elif method == "overlap":
            copy_eval["compare"] = [get_overlap_score(ref_answers[i], test_answer_copies[i]) for i in range(len(copy_eval))]
        elif method == "cosine":
            copy_eval["compare"] = [get_overlap_similarity(ref_answers[i], test_answer_copies[i]) for i in range(len(copy_eval))]
        else:
            print("Unknown method: "+method+"!")

        # Determine prediction: MAX
        max_row = copy_eval.iloc[[copy_eval["compare"].argmax()]]
        max_sim = max_row.iloc[0]["compare"]
        max_pred = max_row.iloc[0]["score2"]
        max_sim_id = max_row.iloc[0]["id1"]
        max_sim_answer = max_row.iloc[0]["text1"]

        # Determine prediction: AVG
        label_avgs = {}
        for label in set(copy_eval["score2"]):
            label_subset = copy_eval[copy_eval["score2"] == label]
            label_avgs[label] = label_subset["compare"].mean()
        avg_pred = max(label_avgs, key=label_avgs.get)
        avg_sim = max(label_avgs.values())

        max_predictions.append(max_pred)
        avg_predictions.append(avg_pred)

        predictions[predictions_index] = {"id": test_answer[id_column], "pred_avg": avg_pred, "sim_score_avg": avg_sim, "pred_max": max_pred, "sim_score_max": max_sim, "most_similar_answer_id": max_sim_id, "most_similar_answer_text": max_sim_answer}
        predictions_index += 1

    copy_test = df_test.copy()
    df_predictions = pd.DataFrame.from_dict(predictions, orient='index')
    df_predictions = pd.merge(copy_test, df_predictions, left_on=id_column, right_on="id")
    df_predictions.to_csv(os.path.join(run_path, "predictions_sim.csv"), index=None)

    return max_predictions, avg_predictions


# print(get_overlap_similarity("this is a test", "this kid likes cats"))
# print(get_overlap_similarity("this is a test", "is this a test"))
# print()
# print(get_overlap_score("this is a test", "this kid likes cats"))
# print(get_overlap_score("this is a test", "is this a test"))
# print()
# print(distance("this is a test", "this kid likes cats"))
# print(distance("this is a test", "is this a test"))
