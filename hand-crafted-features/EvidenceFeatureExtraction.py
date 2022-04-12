import os, re, json
import pandas as pd
import numpy as np
from tokenizer import split_into_sentences
from collections import defaultdict

from readData import readRevisionSentPairAndMetadata
from NgramFeatures import get_ngram_features
from TextualFeatures import get_textual_features
from LanguageFeatures import get_language_features

def getRevisionPairs(revisionName, path):
    datafile = revisionName + 'SentencePairData.xlsx'
    if datafile.endswith(".xlsx"):
        file = os.path.join(path, datafile)
        data = pd.read_excel(file, sheetname=revisionName + 'SentencePairData', encoding="utf-8")
        revisions = list()
        for row in data.iterrows():
            s1 = '' if str(row[1]['S1']).lower() == 'nan' else str(row[1]['S1'])
            s2 = '' if str(row[1]['S2']).lower() == 'nan' else str(row[1]['S2'])
            revisions.append([s1,s2])
    return revisions

def getSentencePairFeatures(revisionName):
    path = '../data/college/'
    datafile =  revisionName + 'SentencePairData.xlsx'

    if datafile.endswith(".xlsx"):
        file = os.path.join(path, datafile)
        data = pd.read_excel(file, sheetname=revisionName + 'SentencePairData', encoding="utf-8")

        spfeatures = []

        for row in data.iterrows():
            id = str(row[1]['ID'])
            s1 = '' if str(row[1]['S1']).lower() == 'nan' else str(row[1]['S1'])
            s2 = '' if str(row[1]['S2']).lower() == 'nan' else str(row[1]['S2'])

            unifname, unifeatures = get_ngram_features(s1, s2)
            textfname, textfeatures = get_textual_features(s1, s2)
            langfname, langfeatures = get_language_features(s1, s2)

            spfeatures.append([id, int(row[1]['index1']), int(row[1]['index2'])] + unifeatures + textfeatures + langfeatures + [str(row[1]['Label'])])

    cols = ['ID', 'index1', 'index2'] + unifname + textfname + langfname + ['Label']
    outputFileName = '../data/output/'+ revisionName + 'SentencePairFeatures.csv'
    df = pd.DataFrame(np.array(spfeatures), columns=cols)
    df.to_csv(outputFileName, sep=',', encoding='utf-8')



def readDataANDFeatures_X_y(revisionName, revisionLabel, path):

    print('Reading Features...')
    # path = '../../Data/eRevise/March2020-Classification/'
    datafile = revisionName + 'SentencePairData.xlsx'

    if datafile.endswith(".xlsx"):
        file = os.path.join(path, datafile)
        data = pd.read_excel(file, sheetname=revisionName + 'SentencePairData', encoding="utf-8")

        X, y, score = [], [], []

        for row in data.iterrows():
            id = str(row[1]['ID'])

            s1 = '' if str(row[1]['S1']).lower() == 'nan' else str(row[1]['S1'])
            s2 = '' if str(row[1]['S2']).lower() == 'nan' else str(row[1]['S2'])
            unifname, unifeatures = get_ngram_features(s1, s2)
            # unifname, unifeatures = get_ngram_features_Article(s1, s2, MVP)

            # e1 = SGDict[id][0]
            # e2 = SGDict[id][1]
            # unifname, unifeatures = get_ngram_features_Article_Essay(s1, s2, e1, e2, MVP)

            # textfname, textfeatures = get_textual_features(s1, s2)
            # langfname, langfeatures = get_language_features(s1, s2)

            # spfeatures.append(
            #     [id, int(row[1]['index1']), int(row[1]['index2'])] + unifeatures + textfeatures + langfeatures + [
            #         str(row[1]['Label'])])

            X.append(unifeatures)# + SGDict[id][2:])
            # X.append(unifeatures + textfeatures + langfeatures + SGDict[id][2:])
            if str(row[1]['Label2']) == revisionLabel:
                y.append(1)
            else:
                y.append(0)

    return np.array(X), np.array(y), score, unifname


if __name__ == "__main__":
    revisionName = 'Evidence'
    RevisionLabels = ['Desirable', 'Relevant', 'Irrelevant', 'Already Exists', 'Non-Text-Based', 'MINIMAL EVIDENCE']

    # X, y, score, fname = readDataANDFeatures_X_y(revisionName, RevisionLabels[0])
    # getSentencePairFeatures(revisionName)

