import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np

from sklearn.model_selection import KFold, train_test_split

from keras import backend
from transformers import BertTokenizer, TFBertModel, BertConfig, TFDistilBertModel, DistilBertTokenizer, DistilBertConfig
import tensorflow as tf

from LSTMmodels import BertModel_bilstm_basic, BertModel_bilstm_context
from LSTMmodels import BertModel_bilstm_feedback, BertModel_bilstm_context_feedback
from LSTMmodels import BertModel_bilstm_basic_forCuDNN, BertModel_bilstm_context_forCuDNN
from augmenter import create_augmented_data, create_augmented_data_context, create_augmented_data_feedback
from augmenter import create_augmented_data_context_feedback, create_augmented_data_balance
from LogRwAugmentation import LogR, load_pretrained_embedding
from util import save_plot

def BertEncoder(sentence_pairs, labels, max_length = 100, encoder_name = 'bert'):
    if encoder_name == 'distilbert':
        bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    bert_inps = bert_tokenizer.batch_encode_plus(sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            truncation=True,
            return_tensors="tf")
    input_ids = np.array(bert_inps["input_ids"], dtype="int64")
    attention_masks = np.array(bert_inps["attention_mask"], dtype="int64")
    token_type_ids = np.array(bert_inps["token_type_ids"], dtype="int64")

    labels = np.array(labels)
    return [input_ids, attention_masks, token_type_ids], labels

def BERT_train_current_fold(X_train, y_train, X_test, y_test, maxlen, fold, params):

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

    if params.isContext and params.isFeedback:
        if params.isSimpleContext:
            max_context_len = 200
        else:
            max_context_len = 512

        if 'MVP' in params.checkpoint_path:
            max_feedback_len = 100
        elif 'Hschool1' in params.checkpoint_path:
            max_feedback_len = 512

        train_context1 = X_train[:, 2]
        train_context2 = X_train[:, 3]
        train_feedback = X_train[:, 4]
        X_train = X_train[:, 0:2]
        train_context1, _ = BertEncoder(train_context1.astype("str"), [], max_length=max_context_len)
        train_context2, _ = BertEncoder(train_context2.astype("str"), [], max_length=max_context_len)
        train_feedback, _ = BertEncoder(train_feedback.astype("str"), [], max_length=max_feedback_len)

        valid_context1 = X_valid[:, 2]
        valid_context2 = X_valid[:, 3]
        valid_feedback = X_valid[:, 4]
        X_valid = X_valid[:, 0:2]
        valid_context1, _ = BertEncoder(valid_context1.astype("str"), [], max_length=max_context_len)
        valid_context2, _ = BertEncoder(valid_context2.astype("str"), [], max_length=max_context_len)
        valid_feedback, _ = BertEncoder(valid_feedback.astype("str"), [], max_length=max_feedback_len)

        test_context1 = X_test[:, 2]
        test_context2 = X_test[:, 3]
        test_feedback = X_test[:, 4]
        X_test = X_test[:, 0:2]
        test_context1, _ = BertEncoder(test_context1.astype("str"), [], max_length=max_context_len)
        test_context2, _ = BertEncoder(test_context2.astype("str"), [], max_length=max_context_len)
        test_feedback, _ = BertEncoder(test_feedback.astype("str"), [], max_length=max_feedback_len)

    elif params.isContext:
        if params.isSimpleContext:
            max_context_len = 200
        else:
            max_context_len = 512
        train_context1 = X_train[:, 2]
        train_context2 = X_train[:, 3]
        X_train = X_train[:, 0:2]
        train_context1, _ = BertEncoder(train_context1.astype("str"), [], max_length=max_context_len)
        train_context2, _ = BertEncoder(train_context2.astype("str"), [], max_length=max_context_len)

        valid_context1 = X_valid[:, 2]
        valid_context2 = X_valid[:, 3]
        X_valid = X_valid[:, 0:2]
        valid_context1, _ = BertEncoder(valid_context1.astype("str"), [], max_length=max_context_len)
        valid_context2, _ = BertEncoder(valid_context2.astype("str"), [], max_length=max_context_len)

        test_context1 = X_test[:, 2]
        test_context2 = X_test[:, 3]
        X_test = X_test[:, 0:2]
        test_context1, _ = BertEncoder(test_context1.astype("str"), [], max_length=max_context_len)
        test_context2, _ = BertEncoder(test_context2.astype("str"), [], max_length=max_context_len)
    elif params.isFeedback:
        if 'MVP' in params.checkpoint_path:
            max_feedback_len = 100
        elif 'Hschool1' in params.checkpoint_path:
            max_feedback_len = 512

        train_feedback = X_train[:, 2]
        X_train = X_train[:, 0:2]
        train_feedback, _ = BertEncoder(train_feedback.astype("str"), [], max_length=max_feedback_len)

        valid_feedback = X_valid[:, 2]
        X_valid = X_valid[:, 0:2]
        valid_feedback, _ = BertEncoder(valid_feedback.astype("str"), [], max_length=max_feedback_len)

        test_feedback = X_test[:, 2]
        X_test = X_test[:, 0:2]
        test_feedback, _ = BertEncoder(test_feedback.astype("str"), [], max_length=max_feedback_len)

    train_data, train_label = BertEncoder(X_train.astype("str"), y_train, max_length=maxlen)
    valid_data, val_label = BertEncoder(X_valid.astype("str"), y_valid, max_length=maxlen)
    test_data, test_label = BertEncoder(X_test.astype("str"), y_test, max_length=maxlen)

    if params.isContext and params.isFeedback:
        model = BertModel_bilstm_context_feedback(maxlen, params.hidden_size, params.learning_rate, max_context_len, max_feedback_len)
        modelpath = "%sSimpleContextFeedback_%s_syn_fold%sb%slr%s" % (params.checkpoint_path, params.model_name, str(fold), params.batch_size, str(params.learning_rate))

    elif params.isContext:
        model = BertModel_bilstm_context(maxlen, params.hidden_size, params.learning_rate)
        modelpath = "%sSimpleContext_%s_syn_fold%sb%slr%s" % (params.checkpoint_path, params.model_name, str(fold), params.batch_size, str(params.learning_rate))

    elif params.isFeedback:
        model = BertModel_bilstm_feedback(maxlen, params.hidden_size, params.learning_rate, max_feedback_len)
        modelpath = "%sFeedback_%s_syn_fold%sb%slr%s" % (params.checkpoint_path, params.model_name, str(fold), params.batch_size, str(params.learning_rate))

    else:
        model = BertModel_bilstm_basic(maxlen, params.hidden_size, params.learning_rate)
        modelpath = "%sBERT_%s_syn_fold%sb%slr%s" % (params.checkpoint_path, params.model_name, str(fold), params.batch_size, str(params.learning_rate))

    print(modelpath)
    print('\nBert Model', model.summary())
    print('')

    isTransfer = False
    if isTransfer and params.istrain:
        checkpoint_path2 = '../checkpoints/elementary/reasoning/' # path of pretrained model
        pretrained_modelpath = "%sBERT_%s_syn_fold%sb%slr%s" % (checkpoint_path2, params.model_name, str(fold), params.batch_size, str(params.learning_rate))
        model.load_weights(pretrained_modelpath + '.h5')
        print('Done loading model weight for transfer learning')


    if params.istrain == True:
        # CREATE CALLBACKS
        earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10 , verbose=0)  # patience = number of epochs with no improvements
        checkpoint = tf.keras.callbacks.ModelCheckpoint(modelpath + '.h5', monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=0)
        callbacks_list = [earlystopper, checkpoint]

        optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        if params.isContext and params.isFeedback:
            history = model.fit(
                [train_data[0], train_data[1], train_context1[0], train_context1[1], train_context2[0], train_context2[1], train_feedback[0], train_feedback[1]], train_label,
                batch_size=params.batch_size, epochs=params.num_epochs, shuffle=True,
                validation_data=([valid_data[0], valid_data[1], valid_context1[0], valid_context1[1], valid_context2[0], valid_context2[1], valid_feedback[0], valid_feedback[1]], val_label),
                callbacks=callbacks_list, verbose=2)

        elif params.isContext:
            history = model.fit([train_data[0], train_data[1], train_context1[0], train_context1[1], train_context2[0], train_context2[1]], train_label,
                                batch_size=params.batch_size, epochs=params.num_epochs, shuffle=True,
                                 validation_data=([valid_data[0], valid_data[1], valid_context1[0], valid_context1[1], valid_context2[0], valid_context2[1]], val_label),
                                callbacks=callbacks_list, verbose=2)

        elif params.isFeedback:
            history = model.fit([train_data[0], train_data[1], train_feedback[0], train_feedback[1]], train_label,
                                batch_size=params.batch_size, epochs=params.num_epochs, shuffle=True,
                                validation_data=([valid_data[0], valid_data[1], valid_feedback[0], valid_feedback[1]], val_label),
                                callbacks=callbacks_list, verbose=2)

        else:
            history = model.fit([train_data[0], train_data[1]], train_label, batch_size=params.batch_size, epochs=params.num_epochs,
                                shuffle=True,
                                validation_data=([valid_data[0], valid_data[1]], val_label), callbacks=callbacks_list,
                                verbose=2)
        save_plot(history, modelpath)
    else:
        print('Loading model weights....')
        model.load_weights(modelpath + '.h5')


    # preds = model.predict(test_data, batch_size=batch_size)
    if params.isContext and params.isFeedback:
        preds = model.predict([test_data[0], test_data[1], test_context1[0], test_context1[1], test_context2[0], test_context2[1], test_feedback[0], test_feedback[1]], batch_size=params.batch_size)

    elif params.isContext:
        preds = model.predict([test_data[0], test_data[1], test_context1[0], test_context1[1], test_context2[0], test_context2[1]], batch_size=params.batch_size)

    elif params.isFeedback:
        preds = model.predict([test_data[0], test_data[1], test_feedback[0], test_feedback[1]], batch_size=params.batch_size)

    else:
        preds = model.predict([test_data[0], test_data[1]], batch_size=params.batch_size)

    pred_labels = [int(round(x[0])) for x in preds]

    # predict test labels
    y_pred = pred_labels #model.predict_classes(test_data, verbose=0)
    y_pred = np.asarray(y_pred).reshape(len(y_pred))
    y_test = np.asarray(y_test).reshape(len(y_test))

    backend.clear_session()
    if params.optimize:
        os.remove(modelpath + '.h5')
    return y_test, y_pred

def train_CV(ID, X, y, maxlen, params):
    if params.embedding_name == 'glove':
        #load before start of cross validation to avoid loading at each fold
        pretrained_embeddings = load_pretrained_embedding(params.embedding_path, emb_dim = params.emb_dim, embedding_name = params.embedding_name)
    elif params.embedding_name == 'distilbert':
        pretrained_embeddings = 'bert'#TFDistilBertModel.from_pretrained('distilbert-base-uncased')

    kf = KFold(n_splits=params.nfolds)  # , random_state=11)
    fold = 1

    y_train_folds = []
    y_test_folds = []
    y_pred_folds = []
    id_pred_folds = []

    for train_index, test_index in kf.split(X):
        print('\n=====================================> Fold : ', fold)

        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        id_train, id_test = np.array(ID)[train_index], np.array(ID)[test_index]
        print('Train data shape: ', X_train.shape, y_train.shape)
        print('Test data shape: ', X_test.shape, y_test.shape)

        y_train_folds.append(y_train) # add y_train before augmentation

        # combine with augmented data
        if params.augment:
            ym_aug = []
            if params.isContext and params.isFeedback:
                X_aug, y_aug = create_augmented_data_context_feedback(X_train, y_train, type=['synonym'])
            elif params.isContext:
                X_aug, y_aug = create_augmented_data_context(X_train, y_train, type=['synonym'])
            elif params.isFeedback:
                X_aug, y_aug = create_augmented_data_feedback(X_train, y_train, type=['synonym'])
            else:
                X_aug, y_aug = create_augmented_data(X_train, y_train, type=['synonym'])
                # X_aug, y_aug = create_augmented_data_balance(X_train, y_train, type=['synonym'])

            X_aug = np.array(X_aug)
            y_aug = np.array(y_aug)
            X_train = np.concatenate((X_train, X_aug), axis=0)
            y_train = np.concatenate((y_train, y_aug), axis=0)

            print('Augmented data shape: ', X_aug.shape, y_aug.shape)
            print('Augmented+Train data shape: ', X_train.shape, y_train.shape)

            des, undes, l = (y_train==1).sum(), (y_train==0).sum(), len(y_train)
            print('Train data balance', des, undes, des / l, undes / l)

        if params.optimize: # extract very few validation data from the train set
            X_train, X_train2, y_train, y_train2 = train_test_split(X_train, y_train, train_size=2000)
            print('Train data shape for parameter optimization: ', X_train.shape, y_train.shape)

        y_pred = []

        if params.model_name == '':
            print('No model name provided ...')
            # raise AttributeError
        elif params.model_name == 'LogR':
            y_test, y_pred = LogR(X_train, y_train, X_test, y_test, pretrained_embeddings, params)
        elif params.model_name == 'BERT-basic':
            model_ext = '-Context' if params.isContext else ('-Feedback' if params.isFeedback else '')
            if params.isContext and params.isFeedback:
                model_ext = '- Context+Feedback'
            print("=========================> Current Model : ", params.model_name, model_ext, " <=========================")

            y_test, y_pred = BERT_train_current_fold(X_train, y_train, X_test, y_test, maxlen, fold, params)

        y_test_folds.append(y_test)
        y_pred_folds.append(y_pred)
        id_pred_folds += list(zip(id_test, y_pred))

        fold += 1

    return y_train_folds, y_test_folds, y_pred_folds, id_pred_folds
















