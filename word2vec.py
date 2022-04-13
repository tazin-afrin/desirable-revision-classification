import numpy as np
import pandas as pd
import os
from nltk import word_tokenize
word_emb_size = 100

def readArticle():
    print('Reading Article...')
    path = '../data/elementary/A Brighter Future.txt'
    f = open(path, "r+")
    contents = f.read()
    return str(contents)

def getRevisionPairs(revisionName, path):
    datafile = revisionName + 'SentencePairData.xlsx'

    if datafile.endswith(".xlsx"):
        file = os.path.join(path, datafile)
        # data = pd.read_excel(file, sheet_name=revisionName + 'SentencePairData', engine='openpyxl')
        data = pd.read_excel(file, sheet_name=revisionName + 'SentencePairData')

        revisions = list()

        for row in data.iterrows():
            s1 = '' if str(row[1]['S1']).lower() == 'nan' else str(row[1]['S1'])
            s2 = '' if str(row[1]['S2']).lower() == 'nan' else str(row[1]['S2'])
            revisions.append([s1,s2])
    return revisions

def readDataLabel(revisionName, revisionLabel, path):

    print('Reading Label...')
    datafile = revisionName + 'SentencePairData.xlsx'

    if datafile.endswith(".xlsx"):
        file = os.path.join(path, datafile)
        # data = pd.read_excel(file, sheet_name=revisionName + 'SentencePairData', engine='openpyxl')
        data = pd.read_excel(file, sheet_name=revisionName + 'SentencePairData')

        y = []

        for row in data.iterrows():
            id = str(row[1]['ID'])

            if str(row[1]['Label2']) == revisionLabel:
                y.append(1)
            else:
                y.append(0)

    return np.array(y)

def get_embedding_matrix(pretrained_embeddings, embedding_dim, num_words, word_index):
    # first create a matrix of zeros, this is our embedding matrix
    embedding_matrix = np.zeros((num_words, embedding_dim))

    # for each word in out tokenizer lets try to find that word in our w2v model
    for word, i in word_index.items():
        embedding_vector = pretrained_embeddings.get(word)
        if embedding_vector is not None:
            # we found the word - add that words vector to the matrix
            embedding_matrix[i] = embedding_vector
        else:
            # doesn't exist, assign a random vector
            embedding_matrix[i] = np.random.randn(embedding_dim)
            # embedding_matrix[i] = np.zeros(embedding_dim)
    # print("************ USING ZEROS FOR OOV**************")

    return embedding_matrix

def get_GLOVE_word2vec(glove_path, word_emb_size):
    print("Loading glove.6B.{}d...".format(word_emb_size))
    glove_file = "{}glove.6B.{}d.txt".format(glove_path, word_emb_size)
    word2vec_dict = {}
    with open(glove_file, 'r+', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            # vector = np.asarray(values[1:])
            word2vec_dict[word] = vector
    print('Finished Loading glove')
    return word2vec_dict


# aggregate vectors matrix for a long text word by word
# substitute unknown word vectors with zeros
def inferText2Vec(text, word2vec_dict):
    vec = list()
    text = word_tokenize(text)
    if text == []:
        return np.zeros(word_emb_size)

    for word in text:
        if word in word2vec_dict:
            vec.append(word2vec_dict[word])
        else:
            vec.append(np.zeros(word_emb_size))

    vec = np.mean(np.array(vec), axis=0)

    return vec

# aggregate vectors matrix for a long text word by word
# this is the function used with augmented data in later experiments
def inferText2Vec_(text, word2vec_dict):
    vec = list()
    if text == []:
        return np.zeros(word_emb_size)

    for word in text:
        vec.append(word2vec_dict[word])
        # else:
        #     vec.append(np.zeros(word_emb_size))
    # print(vec)
    vec = np.mean(np.array(vec), axis=0)

    return vec


# This is the main function used
def inferWord2VecFeatures(revisionName, path):
    word2vec_dict = get_GLOVE_word2vec("../glove/", word_emb_size)

    revisions = getRevisionPairs(revisionName, path)
    article = readArticle()

    articlevector = inferText2Vec(article, word2vec_dict)

    test_vectors = list()
    # infer test vectors
    for s1,s2 in revisions:
        s1vec = inferText2Vec(s1, word2vec_dict)
        s2vec = inferText2Vec(s2, word2vec_dict)

        # This is the original mean vector calculation published in the BEA'20 paper
        vec = np.array([s1vec, s2vec, articlevector])
        vec = np.mean(vec,axis=0)

        # Try other way of creating the vector from w2v
        # vec = np.array([s1vec, s2vec])
        # vec = np.mean(vec,axis=0)

        test_vectors.append(vec)

    return np.array(test_vectors)



# this is the function used with augmented data in later experiments
# without article
def inferWord2VecFeatures_(X_train, X_test, word2vec_dict):

    train_vectors = list()
    test_vectors = list()

    # infer test vectors
    for i in range(len(X_train[0])):
        s1 = X_train[0][i]
        s2 = X_train[1][i]
        s1vec = inferText2Vec_(s1, word2vec_dict)
        s2vec = inferText2Vec_(s2, word2vec_dict)

        vec = np.array([s1vec, s2vec])
        vec = np.mean(vec,axis=0)

        train_vectors.append(vec)

    for i in range(len(X_test[0])):
        s1 = X_test[0][i]
        s2 = X_test[1][i]

        s1vec = inferText2Vec_(s1, word2vec_dict)
        s2vec = inferText2Vec_(s2, word2vec_dict)

        vec = np.array([s1vec, s2vec])
        vec = np.mean(vec,axis=0)

        test_vectors.append(vec)

    return np.array(train_vectors), np.array(test_vectors)

if __name__=="__main__":

    X = inferWord2VecFeatures()
    print(len(X), len(X[0]), X[0])