import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.model_selection import KFold, train_test_split

from keras import backend

from augmenter import create_augmented_data
from MultitaskLSTM import  multitask_model, BertEncoder, BertModel_bilstm_basic


def train_CV(datasets, params):
    results = defaultdict(defaultdict)
    kf = KFold(n_splits=params.nfolds)  # , random_state=11)

    fold_splits = {}
    for dataName in params.dataNames:
        X = datasets[dataName]['X']
        # create folds for each dataset
        fold_splits[dataName] = list(kf.split(X))

        # create variables to save info from each fold
        results[dataName]['y_train_folds'] = []
        results[dataName]['y_test_folds'] = []
        results[dataName]['y_pred_folds'] = []
        results[dataName]['id_test_folds'] = []
        results[dataName]['id_pred_folds'] = [] # to zip id and pred

    for fold in range(params.nfolds):
        # print('\n=====================================> Fold : ', fold+1)

        augmented_data = defaultdict(defaultdict)
        # create augmented data within the fold
        # loop over datasets in each fold to create augmented data
        for dataName in params.dataNames:
            # get X and y for each dataset
            X = datasets[dataName]['X']
            y = datasets[dataName]['y']
            ID = datasets[dataName]['ID']

            # from X and y extract the data for the current fold
            train_index, test_index = fold_splits[dataName][fold]
            X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
            id_train, id_test = np.array(ID)[train_index], np.array(ID)[test_index]
            print(dataName, 'data shape Train:', X_train.shape, y_train.shape, 'Test:', X_test.shape, y_test.shape)

            # save train,test,id info for the current fold
            results[dataName]['y_train_folds'].append(y_train)
            results[dataName]['y_test_folds'].append(y_test)
            results[dataName]['id_test_folds'].append(id_test)

            # combine train data with augmented data
            if params.augment:
                X_aug, y_aug = create_augmented_data(X_train, y_train, type=['synonym'])
                X_train = np.concatenate((X_train, np.array(X_aug)), axis=0)
                y_train = np.concatenate((y_train, np.array(y_aug)), axis=0)

                X_train = X_train[0:5120,:]
                y_train = y_train[0:5120]
                print('Augmented+Train data shape: ', X_train.shape, y_train.shape)

                des, undes, l = (y_train == 1).sum(), (y_train == 0).sum(), len(y_train)
                print('Train data balance', des, undes, des / l, undes / l)
                # print(des, undes, des / l, undes / l)
                # print('')

                augmented_data[dataName]['X_train'] = X_train
                augmented_data[dataName]['y_train'] = y_train
                augmented_data[dataName]['X_test'] = X_test
                augmented_data[dataName]['y_test'] = y_test

        # train for the current fold
        y_pred_dict = {}
        if params.model_name == '':
            print('No model name provided ...')
            # raise AttributeError
        elif params.model_name == 'BERT-basic':
            model_ext = '-Context' if params.isContext else ('-Feedback' if params.isFeedback else '')
            print("=========================> Current Model : ", params.model_name, model_ext, " <=========================")
            y_pred_dict = train_model(augmented_data, fold, params)

        # save predicted result
        for dataName in params.dataNames:
            y_pred = y_pred_dict[dataName]
            results[dataName]['y_pred_folds'].append(y_pred)
            id_test = results[dataName]['id_test_folds'][-1]
            results[dataName]['id_pred_folds'] += list(zip(id_test, y_pred))


        fold += 1

    return results

def save_plot_multitask(history, modelpath, params):
    for dataName in params.dataNames:
        plt.plot(history[dataName])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(params.dataNames, loc='upper right')
    plt.savefig(modelpath + '.png')
    plt.close()

def train_model(augmented_data, fold, params):
    # create multitask model for each data with shared layer
    if params.isMultitask:
        models = multitask_model(params)
        for dataName in params.dataNames:
            print(models[dataName].summary())
    else:
        model = BertModel_bilstm_basic(params)
        print(model.summary())

    # encode data for each dataset
    encoded_data = defaultdict(defaultdict)
    for dataName in params.dataNames:
        X_train = augmented_data[dataName]['X_train']
        y_train = augmented_data[dataName]['y_train']
        X_test = augmented_data[dataName]['X_test']
        y_test = augmented_data[dataName]['y_test']

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

        train_data, train_label = BertEncoder(X_train.astype("str"), y_train, max_length=params.maxlen)
        valid_data, val_label = BertEncoder(X_valid.astype("str"), y_valid, max_length=params.maxlen)
        test_data, test_label = BertEncoder(X_test.astype("str"), y_test, max_length=params.maxlen)

        encoded_data[dataName]['train_data'] = train_data
        encoded_data[dataName]['train_label'] = train_label
        encoded_data[dataName]['valid_data'] = valid_data
        encoded_data[dataName]['val_label'] = val_label
        encoded_data[dataName]['test_data'] = test_data
        encoded_data[dataName]['test_label'] = test_label

        batches = len(train_data[0])


    # checkpoint_path = params.data_path + 'MultiDataTask/' + params.checkpoint_path + params.revision_name + '/'+params.model_folder+'/'
    checkpoint_path = '../../Data2/RERAnalysis/' + 'MultiDataTask/' + params.checkpoint_path + params.revision_name + '/'+params.model_folder+'/'
    modelpath = "%s%s_syn_fold%sb%slr%s" % (checkpoint_path, params.model_name, str(fold + 1), params.batch_size, str(params.learning_rate))
    # modelpath = "%sUnionBL_%s_syn_fold%sb%slr%s" % (checkpoint_path, params.model_name, str(fold + 1), params.batch_size, str(params.learning_rate))
    # modelpath = "%sUnionBaseline_%s_syn_fold%sb%slr%s" % (checkpoint_path, params.model_name, str(fold + 1), params.batch_size, str(params.learning_rate))
    print(modelpath)


    if params.istrain == True and params.isMultitask:
        # NO CALLBACKS are created

        train_loss = defaultdict(list)
        # withing each epoch, train using each dataset
        for epoch in range(1,params.num_epochs+1):
            for b in range(0, batches, params.batch_size):
                for dataName in params.dataNames:

                    train_data0 = encoded_data[dataName]['train_data'][0][b:b+params.batch_size]
                    train_data1 = encoded_data[dataName]['train_data'][1][b:b+params.batch_size]
                    train_label = encoded_data[dataName]['train_label'][b:b+params.batch_size]
                    valid_data0 = encoded_data[dataName]['valid_data'][0]
                    valid_data1 = encoded_data[dataName]['valid_data'][1]
                    val_label = encoded_data[dataName]['val_label']

                    train_history = models[dataName].fit([train_data0, train_data1], train_label, batch_size=params.batch_size, epochs=1,
                                        shuffle=True,
                                        validation_data=([valid_data0, valid_data1], val_label),
                                        verbose=1)

                    if b+params.batch_size == batches:
                        # print(b+params.batch_size,batches)
                        train_loss[dataName].append(train_history.history['loss'])
        # save model and plot
        for dataName in params.dataNames:
            print(modelpath+'_'+dataName+'.h5')
            models[dataName].save_weights(modelpath+'_'+dataName+'.h5')
        save_plot_multitask(train_loss, modelpath, params)
        # exit(0)

    elif params.istrain == True and params.isMultitask == False:
        train_loss = defaultdict(list)
        # withing each epoch, train using each dataset
        for epoch in range(1, params.num_epochs + 1):
            for b in range(0, batches, params.batch_size):
                # print(b, b + params.batch_size, batches)
                for dataName in params.dataNames:
                    # print('Epoch:', epoch, 'Data: ', dataName)
                    train_data0 = encoded_data[dataName]['train_data'][0][b:b + params.batch_size]
                    train_data1 = encoded_data[dataName]['train_data'][1][b:b + params.batch_size]
                    train_label = encoded_data[dataName]['train_label'][b:b + params.batch_size]
                    valid_data0 = encoded_data[dataName]['valid_data'][0]
                    valid_data1 = encoded_data[dataName]['valid_data'][1]
                    val_label = encoded_data[dataName]['val_label']

                    train_history = model.fit([train_data0, train_data1], train_label,
                                                         batch_size=params.batch_size, epochs=1,
                                                         shuffle=True,
                                                         validation_data=([valid_data0, valid_data1], val_label),
                                                         verbose=2)
                    if b + params.batch_size == batches:
                        train_loss[dataName].append(train_history.history['loss'])

        # save model and plot
        print(modelpath + '.h5')
        model.save_weights(modelpath + '.h5')
        save_plot_multitask(train_loss, modelpath, params)
    elif params.istrain == False and params.isMultitask:
        print('Loading model weights....')
        for dataName in params.dataNames:
            print(modelpath+'_'+dataName+'.h5')
            continue
            models[dataName].load_weights(modelpath+'_'+dataName+'.h5')

    elif params.istrain == False and params.isMultitask == False:
        print('Loading model weights....')
        model.load_weights(modelpath + '.h5')

    y_pred_dict = {}
    for dataName in params.dataNames:
        test_data = encoded_data[dataName]['test_data']
        test_label = encoded_data[dataName]['test_label']
        if params.isMultitask:
            preds = models[dataName].predict([test_data[0], test_data[1]], batch_size=params.batch_size)
        else:
            preds = model.predict([test_data[0], test_data[1]], batch_size=params.batch_size)

        pred_labels = [int(round(x[0])) for x in preds]

        # predict test labels
        y_pred = pred_labels
        y_pred = np.asarray(y_pred).reshape(len(y_pred))

        y_pred_dict[dataName] = y_pred

    backend.clear_session()
    return y_pred_dict
