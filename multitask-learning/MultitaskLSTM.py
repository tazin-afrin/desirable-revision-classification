import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertConfig, TFDistilBertModel, DistilBertTokenizer, DistilBertConfig

def multitask_model(params):
    Models = {}
    max_len = params.maxlen
    hidden_size = params.hidden_size

    # print('Load transformer model ...')
    config = BertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    transformer_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)

    input_ids_in = tf.keras.layers.Input(shape=(max_len,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(max_len,), name='masked_token', dtype='int32')

    embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
    shared_layer = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=0.1), name='shared_LSTM')(embedding_layer)
    shared_layer = tf.keras.layers.GlobalMaxPool1D()(shared_layer)

    for modelName in params.dataNames:
        print('modelName:', modelName)
        output = shared_layer
        X = tf.keras.layers.Dense(hidden_size, activation='relu', name=modelName+'_relu')(output)
        X = tf.keras.layers.Dropout(0.2, name=modelName+'_dropout')(X)
        X = tf.keras.layers.Dense(1, activation='sigmoid', name=modelName+'_sigmoid')(X)
        model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=X, name=modelName+'_model')
        for layer in model.layers[:3]:
            print(layer.name)
            layer.trainable = False

        optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        Models[modelName] = model

    return Models

def BertModel_bilstm_basic(params):
    max_len = params.maxlen
    hidden_size = params.hidden_size

    config = BertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    transformer_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)

    input_ids_in = tf.keras.layers.Input(shape=(max_len,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(max_len,), name='masked_token', dtype='int32')

    embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
    X = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding_layer)
    X = tf.keras.layers.GlobalMaxPool1D()(X)
    X = tf.keras.layers.Dense(hidden_size, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid')(X)
    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=X)

    for layer in model.layers[:3]:
        print(layer.name)
        layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


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
