import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score


def t_test(list1, list2):
    ttest = stats.ttest_ind(list1, list2)
    return ttest[1]

def MajorityClass(y_train, y, maj = 1):
    x = np.array(y, dtype='|S4')
    y_test = x.astype(np.int)

    l = len(y_train)/2.0
    maj = 1 if sum(y_train) > l else 0
    y_pred = [maj for i in range(len(y_test))]

    return y_pred

def evaluate_folds(y_train_folds, y_test_folds, y_pred_folds):
    qwk = []
    acc1, prec1, rec1, f11, maj1 = ([] for i in range(5))
    acc0, prec0, rec0, f10, maj0 = ([] for i in range(5))
    acc, prec, rec, f1, pbase, rbase, f1base = ([] for i in range(7))

    for i in range(len(y_test_folds)):
        y_train = y_train_folds[i]
        y_test = y_test_folds[i]
        y_pred = y_pred_folds[i]


        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        prec.append(p)
        rec.append(r)
        f1.append(f)

        y_baseline = MajorityClass(y_train, y_test)
        # y_baseline = random_algorithm(y_test)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_baseline, average='macro')
        pbase.append(p)
        rbase.append(r)
        f1base.append(f)

        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
        prec1.append(p)
        rec1.append(r)
        f11.append(f)

        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=0, average='binary')
        prec0.append(p)
        rec0.append(r)
        f10.append(f)

        qk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
        if math.isnan(qk):
            continue
        qwk.append(qk)


    print('base: ', round(np.array(pbase).mean(),3), round(np.array(rbase).mean(),3), round(np.array(f1base).mean(),3))
    print('clf-avg:', round(np.array(prec).mean(),3), round(np.array(rec).mean(),3), round(np.array(f1).mean(),3))
    print('sig: ', round(t_test(prec, pbase),3), round(t_test(rec, rbase),3), round(t_test(f1, f1base),3))

    print('Label:1 ', round(np.array(prec1).mean(), 3), round(np.array(rec1).mean(), 3), round(np.array(f11).mean(), 3))
    print('Label:0 ', round(np.array(prec0).mean(), 3), round(np.array(rec0).mean(), 3), round(np.array(f10).mean(), 3))

    print('prec = ', prec)
    print('rec = ', rec)
    print('f1 = ', f1)
    print('')
    print('prec1 = ', prec1)
    print('rec1 = ', rec1)
    print('f11 = ', f11)
    print('prec0 = ', prec0)
    print('rec0 = ', rec0)
    print('f10 = ', f10)
    print('')
    print('qwk = ', qwk)
    print('QWK: ', round(np.array(qwk).mean(),3))

def getScorebyID(data):
    score_by_id = []
    datafile = '../data/'+data+'/score.xlsx'

    if datafile.endswith(".xlsx"):
        data = pd.read_excel(datafile)

        if 'ImprovementScore' in data.columns:
            col = 'ImprovementScore'
        elif 'TotalScoreChange' in data.columns:
            col = 'TotalScoreChange'
        else:
            col = 'TotalScoreChangeDirection'

        for idx, row in data.iterrows():
            score_by_id.append([row['ID'], row[col]])

    return score_by_id


def format_corr(corr):
    c, p = corr
    star = ''
    if p < 0.001:
        star = '***'
    elif p < 0.01:
        star = '**'
    elif p <= 0.05:
        star = '*'

    return str(round(c,3))+star


def extrinsic_evaluation(y_pred_by_id, score_by_id, adm):
    y_pred_add, y_pred_delete, y_pred_modify, y_pred_total, scores = [], [], [], [], []

    output_des = []
    output_undes = []

    for id1, s in score_by_id:
        desirable = [0,0,0,0] # add, del, mod, total
        undesirable = [0,0,0,0]
        for i in range(len(y_pred_by_id)):
            id2, label = y_pred_by_id[i]
            # print(id1, id2)
            if id2 == id1:
                if label == 1:
                    if adm[i] == 1:
                        desirable[0] += 1
                    elif adm[i] == 2:
                        desirable[1] += 1
                    else:
                        desirable[2] += 1
                    desirable[3] += 1
                else:
                    if adm[i] == 1:
                        undesirable[0] += 1
                    elif adm[i] == 2:
                        undesirable[1] += 1
                    else:
                        undesirable[2] += 1
                    undesirable[3] += 1
        scores.append(int(s))
        output_des.append(desirable)
        output_undes.append(undesirable)
        # print(id1, s, desirable, undesirable)

    scores = np.array(scores)
    output_des = np.array(output_des)
    output_undes = np.array(output_undes)


    print('Extrinsic Evaluation : \n \t Corr \t p-value for desirable and undesirable')
    print('add', stats.pearsonr(output_des[:, 0], scores), stats.pearsonr(output_undes[:, 0], scores))
    print('del', stats.pearsonr(output_des[:, 1], scores), stats.pearsonr(output_undes[:, 1], scores))
    print('mod', stats.pearsonr(output_des[:, 2], scores), stats.pearsonr(output_undes[:, 2], scores))
    print('tot', stats.pearsonr(output_des[:, 3], scores), stats.pearsonr(output_undes[:, 3], scores))

    print('Extrinsic Evaluation : \n \t Corr for add/del/mod/total on each column \n desirable and undesirable on each row')
    ca = stats.pearsonr(output_des[:, 0], scores)
    cd = stats.pearsonr(output_des[:, 1], scores)
    cm = stats.pearsonr(output_des[:, 2], scores)
    ct = stats.pearsonr(output_des[:, 3], scores)
    print(format_corr(ca), format_corr(cd), format_corr(cm), format_corr(ct))
    ca = stats.pearsonr(output_undes[:, 0], scores)
    cd = stats.pearsonr(output_undes[:, 1], scores)
    cm = stats.pearsonr(output_undes[:, 2], scores)
    ct = stats.pearsonr(output_undes[:, 3], scores)
    print(format_corr(ca), format_corr(cd), format_corr(cm), format_corr(ct))
