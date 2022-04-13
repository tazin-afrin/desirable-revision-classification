import pandas as pd
from keras.preprocessing.text import Tokenizer

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from word2vec import get_GLOVE_word2vec, get_embedding_matrix, inferWord2VecFeatures_

def load_pretrained_embedding(embedding_path,
        emb_dim = 100,
        embedding_name = 'glove'):
    if embedding_name == 'glove':
        emb = get_GLOVE_word2vec(embedding_path, emb_dim)
    else:
        emb = []
    return emb

def prepare_fold_for_training_LogR(X_train, y_train, X_test, y_test, pretrained_embeddings):

    df_train = pd.DataFrame(X_train, columns=['S1', 'S2'])
    df_test = pd.DataFrame(X_test, columns=['S1', 'S2'])

    tokenizer = Tokenizer(num_words=None, split=' ', oov_token='<unk>', filters=' ')
    tokenizer.fit_on_texts(df_train['S1'])
    tokenizer.fit_on_texts(df_train['S2'])
    num_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    print('Vocab size : ', num_words)

    emb_mat = get_embedding_matrix(pretrained_embeddings, 100, num_words, word_index)
    print('Embedding matrix length: ', len(emb_mat))

    # this takes our sentences and replaces each word with an integer
    S1_train = tokenizer.texts_to_sequences(df_train['S1'])
    S2_train = tokenizer.texts_to_sequences(df_train['S2'])
    S1_test = tokenizer.texts_to_sequences(df_test['S1'])
    S2_test = tokenizer.texts_to_sequences(df_test['S2'])

    X_train, X_test = inferWord2VecFeatures_([S1_train, S2_train], [S1_test, S2_test], emb_mat)

    print('Final train and test data shape: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_train, X_test


def LogR(X_train, y_train, X_test, y_test, pretrained_embeddings, params):
    if params.embedding_name == 'glove':
        X_train, X_test = prepare_fold_for_training_LogR(X_train, y_train, X_test, y_test, pretrained_embeddings)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return y_test, y_pred