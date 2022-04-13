import argparse, logging, sys
import numpy as np

from data_prepare import read_multidata
from Multitask_trainer import train_CV
from evaluator import evaluate_folds, extrinsic_evaluation, getScorebyID

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7";

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

np.random.seed(11)

loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

######################################
# multitask with different datasets
# predict one type desirable revision
######################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RER model")

    parser.add_argument('--dind', type=int, help='data index', default=0)
    # i = 1
    parser.add_argument('--data_path', type=str, help='data file', default='../data/')  # +data[i]+'/')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint directory', default='checkpoints/')


    parser.add_argument('--embedding_name', type=str, choices=['glove', 'BERT'], default='BERT',
                        help='Word embedding type, glove, word2vec, senna or random')
    parser.add_argument('--embedding_path', type=str, default='../glove/',
                        help='Pretrained embedding path')
    parser.add_argument('--emb_dim', type=int, default=100,
                        help='Only useful when embedding is randomly initialised')

    parser.add_argument('--model_name', type=str, choices=['BERT-basic'],
                        help='NN classifier model name', default='BERT-basic')
    parser.add_argument('--model_folder', type=str, help='multitask vs baseline union single task', default='ArgH1H2MVP')


    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of instances in each batch')
    parser.add_argument('--hidden_size', type=int, default=64, help='Num of hidden units in recurrent layer')
    parser.add_argument('--optimizer', choices=['adam', 'sgd', 'adagrad', 'rmsprop'], help='Optimizer', default='adam')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.02, help='Initial learning rate')

    parser.add_argument('--revision_name', type=str, choices=['Evidence', 'Reasoning', 'Multitask'],
                        help='Reasoning or Evidence revisions', default='Reasoning')
    parser.add_argument('--revision_label', type=str, help='RER label to classify', default='Desirable')
    # parser.add_argument('--revision_categories', type=str, choices=['all', 'des'], help='Desirable or not VS each RER category classification', default='des')
    parser.add_argument('--nfolds', type=int, default=10, help='number of folds in cross-validation')
    parser.add_argument('--augment', type=bool, default=True, help='augment data or not')
    parser.add_argument('--istrain', type=bool, default=False, help='train model or not')
    parser.add_argument('--optimize', type=bool, default=False, help='optimize model or not')
    parser.add_argument('--isContext', type=bool, default=False, help='train model with context or not')
    parser.add_argument('--isFeedback', type=bool, default=False, help='train model with feedback or not')
    parser.add_argument('--isMultitask', type=bool, default=True, help='Multitask training or not')

    params = parser.parse_args()

    data = ['college', 'Hschool1', 'Hschool2', 'elementary']
    datasets = read_multidata(data, params)

    params.maxlen = max([datasets[data]['maxlen'] for data in datasets])
    params.dataNames = data


    results = train_CV(datasets, params)

    for dataName in params.dataNames:
        print('\nEvaluations on '+dataName+':')
        y_train_folds = results[dataName]['y_train_folds']
        y_test_folds = results[dataName]['y_test_folds']
        y_pred_folds = results[dataName]['y_pred_folds']
        id_pred_folds = results[dataName]['id_pred_folds']

        evaluate_folds(y_train_folds, y_test_folds, y_pred_folds)
        score_by_id = getScorebyID(dataName)
        extrinsic_evaluation(id_pred_folds, score_by_id, datasets[dataName]['adm'])





