import sys
import os
import logging
import torch
import torch.nn as nn
import torch.utils.data as Data
import time

from learning_curve.npcr import data_prepare
from learning_curve.npcr.evaluator_core import Evaluator_opti
from learning_curve.npcr.networks.core_networks import npcr_model


# TODO: Drop base model?
def train_npcr_orig(run_path, df_train, df_test, df_val, prompt_name_column="prompt", answer_column="text", target_column="label", id_column="id", base_model="all-MiniLM-L6-v2", num_epochs=20, batch_size=6, learning_rate=0.00001):

    prompt_id_set=list(df_train[prompt_name_column].unique())
    if len(prompt_id_set)>1:
        print("Impure training data covers more than one prompt!")
        sys.exit(0)

    prompt_id=prompt_id_set[0]
    USE_CHAR = False
    npcr_group = False
    checkpoint_dir = os.path.join(run_path, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    mode = "att"
    nbfilters = 100

    datapaths = [df_train, df_val, df_test]

    modelname = "rank_core_bert.prompt%s.pt" % prompt_id
    imgname = "sent_hiclstm-%s.prompt%s.%sfilters.bs%s.png" % (mode, prompt_id, nbfilters, batch_size)

    if USE_CHAR:
        modelname = 'char_' + modelname
        imgname = 'char_' + imgname

    (X_train,Y_train,mask_train),(X_dev,Y_dev,mask_dev),(X_test,Y_test,mask_test),max_num = \
        data_prepare.prepare_sentence_data(datapaths,prompt_id)

    # train_x0, train_x1, train_y, dev_x,dev_y, dev_yy,test_x,test_y,test_yy = data_prepare.data_pre_con(X_train,
    # Y_train, X_dev, Y_dev, X_test, Y_test,prompt_id)
    if not npcr_group:
        train_x0, train_x1, train_y, features_dev, dev_y_example, dev_y_goal, features_test, test_y_example, test_y_goal = data_prepare.data_pre_opti_ampap(X_train, Y_train, X_dev, Y_dev, X_test, Y_test,prompt_id)
    else:
        print("Grouping not adapted to LC setting yet!")
        sys.exit(0)
        # train_x0, train_x1, train_y, features_dev, dev_y_example, dev_y_goal, features_test, test_y_example, test_y_goal = data_prepare.data_pre_group(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, prompt_id, example_size)

    logging.info("----------------------------------------------------")

    model = npcr_model(512)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()

    # How many pairs were generated for each dev and test instance? (= As many as there are training instances)
    example_size=X_train.shape[0]
    evl = Evaluator_opti(prompt_id, USE_CHAR, checkpoint_dir, modelname, features_dev, dev_y_example, dev_y_goal, features_test, test_y_example, test_y_goal, example_size)

    logging.info("Train model")

    loss_fn = nn.MSELoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # AdamW
    # optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

    torch_dataset = Data.TensorDataset(train_x0, train_x1, torch.Tensor(train_y))

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    for ii in range(num_epochs):
        logging.info('Epoch %s/%s' % (str(ii), num_epochs))
        start_time = time.time()
        for step, (batch_x0, batch_x1, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            Y_predict = model(batch_x0.cuda(), batch_x1.cuda())
            loss = loss_fn(Y_predict.squeeze(), batch_y.squeeze().cuda())
            print('epoch:', ii, 'step:', step, 'loss:', loss.item())
            loss.backward()
            optimizer.step()

        tt_time = time.time() - start_time
        logging.info("Training one epoch in %.3f s" % tt_time)

        model.eval()
        with torch.no_grad():
            evl.evaluate(model, ii, True)
        model.train()

        ttt_time = time.time() - start_time - tt_time
        logging.info("Evaluate one time in %.3f s" % ttt_time)

    pred = evl.print_final_info()
    return pred
