from learning_curve.train_sbert import eval_sbert
import pandas as pd

def get_predictions_pretrained(run_path, df_train, df_val, df_test, model, id_column="id", answer_column="text", target_column="label"):

    df_test['embedding'] = df_test[answer_column].apply(model.encode)

    df_ref = pd.concat([df_val, df_train])
    df_ref['embedding'] = df_ref[answer_column].apply(model.encode)

    return eval_sbert(run_path=run_path, df_test=df_test, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)