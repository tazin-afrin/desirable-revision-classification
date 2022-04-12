import numpy as np
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

from EvidenceFeatureExtraction import readDataANDFeatures_X_y
from word2vec import inferWord2VecFeatures, readDataLabel


def t_test(list1, list2):
    ttest = stats.ttest_ind(list1, list2)
    return ttest[1]


def CV_FS(X,y, nfolds, classifier = 'LogR', featurenames=[]):
    # random.seed = 11
    qwk = []
    acc1, prec1, rec1, f11, maj1 = ([] for i in range(5))
    acc0, prec0, rec0, f10, maj0 = ([] for i in range(5))
    acc, prec, rec, f1, pbase, rbase, f1base = ([] for i in range(7))
    Y_pred = []

    fold = 1
    kf = KFold(n_splits=nfolds, random_state=11)
    print(kf.n_splits, len(X))
    for train_index, test_index in kf.split(X):
        print('Fold : ', fold)
        fold += 1

        X_train, X_test = np.asarray(X[train_index]), np.asarray(X[test_index])
        y_train, y_test = np.asarray(y[train_index]), np.asarray(y[test_index])

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # X_train, X_test = SelectFeatures(X_train, y_train, X_test, featurenames)

        if classifier == 'LogR':
            clf = LogisticRegression(random_state=0)
        elif classifier == 'SVM':
            clf = SVC(kernel='rbf')

        # clf = SVC(C = 10.0, gamma=0.1, kernel='rbf')
        # clf = RandomForestClassifier(random_state=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        Y_pred += list(y_pred)

        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        prec.append(p)
        rec.append(r)
        f1.append(f)

        y_baseline = MajorityClass(y_train, y_test)
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

        print(y_test)


    print('base: ', round(np.array(pbase).mean(),3), round(np.array(rbase).mean(),3), round(np.array(f1base).mean(),3))
    print('clf-avg:', round(np.array(prec).mean(),3), round(np.array(rec).mean(),3), round(np.array(f1).mean(),3))
    print('sig: ', round(t_test(prec, pbase),3), round(t_test(rec, rbase),3), round(t_test(f1, f1base),3))

    print('Label:1 ', round(np.array(prec1).mean(), 3), round(np.array(rec1).mean(), 3), round(np.array(f11).mean(), 3))
    print('Label:0 ', round(np.array(prec0).mean(), 3), round(np.array(rec0).mean(), 3), round(np.array(f10).mean(), 3))

    print(qwk)
    print('QWK: ', round(np.array(qwk).mean(),3))

    return Y_pred


def MajorityClass(y_train, y, maj = 1):
    x = np.array(y, dtype='|S4')
    y_test = x.astype(np.int)

    l = len(y_train)/2.0
    maj = 1 if sum(y_train) > l else 0
    y_pred = [maj for i in range(len(y_test))]

    return y_pred


def save_plot(model_history, save_dir, fold):
    plt.plot(model_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    # plt.show()
    plt.savefig(save_dir+'convergence_fold_'+str(fold)+'.png')
    plt.close()


if __name__ == "__main__":
    path = '../data/college/'
    nfolds = 10
    revisionName = ['Evidence', 'Reasoning']
    # RevisionLabels = ['LCE', 'GENERIC', 'PARAPHRASE', 'NOT LCE', 'COMMENTARY', 'MINIMAL REASONING']
    # RevisionLabels = ['LCE', 'PARAPHRASE', 'NOT LCE', 'COMMENTARY']
    # RevisionLabels = ['Desirable', 'Undesirable']
    # RevisionLabels = ['Relevant', 'Irrelevant']#, 'Non-Text-Based', 'Repeat', 'MINIMAL EVIDENCE']
    RevisionLabels = 'Desirable'
    classifiers = ['LogR', 'SVM']

    ishandcrafted = False

    for i in range(len(revisionName)):
        print('\n*******', revisionName[i], RevisionLabels, '*******\n')

        if ishandcrafted:
            # this is for extracting unigram features
            X, y, score, featureNames = readDataANDFeatures_X_y(revisionName[i], RevisionLabels, path)
            print('Number of uniFeatures: ', len(featureNames))
            print(featureNames)
        else:
            # this is for word2vec features
            X = inferWord2VecFeatures(revisionName[i], path)
            y = readDataLabel(revisionName[i], RevisionLabels, path)

            features = len(X[0])
            print('Number of Current Features: ', features)

        for k in range(0,len(classifiers)-1):
            classifier = classifiers[k]
            print('Current classifier: ', classifier)

            # classifier for LogR
            # call cross validation with feature selection
            y_pred = CV_FS(X, y, nfolds, classifier)
