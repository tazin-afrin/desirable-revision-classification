from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import  numpy as np

def important_features(clf, featurenames):
    if featurenames == []:
        return
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")

    for f in range(len(featurenames)):
        print("%d. feature %s (%f)" % (f + 1, featurenames[indices[f]], importances[indices[f]]))

# tested but not used
def SelectFeatures(X_train, y_train, X_test, featurenames=[]):
    clf = ExtraTreesClassifier(random_state=11).fit(X_train, y_train)
    important_features(clf, featurenames)
    model = SelectFromModel(clf, prefit=True)

    X_train = model.transform(X_train)
    X_test = model.transform(X_test)

    print('Number of features after selection', X_train.shape[1])
    return X_train, X_test