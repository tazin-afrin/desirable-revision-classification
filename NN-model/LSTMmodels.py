from transformers import TFBertModel, TFBertForSequenceClassification, BertConfig
import tensorflow as tf

# This is original baseline model used for MVP1 Evidence then for all revision and all data
def BertModel_bilstm_basic(max_len, hidden_size, learning_rate):
    config = BertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    transformer_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    print('loaded transformer')
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

    return model

def BertModel_bilstm_basic_forCuDNN(max_len, hidden_size, learning_rate):
    config = BertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    transformer_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    print('loaded transformer')
    input_ids_in = tf.keras.layers.Input(shape=(max_len,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(max_len,), name='masked_token', dtype='int32')

    embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
    X = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=0.1))(embedding_layer)
    X = tf.keras.layers.GlobalMaxPool1D()(X)
    X = tf.keras.layers.Dense(hidden_size, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid')(X)
    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=X)

    for layer in model.layers[:3]:
        print(layer.name)
        layer.trainable = False

    return model


def BertModel_bilstm_context_forCuDNN(max_len, hidden_size, learning_rate):
    config = BertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    transformer_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)

    input_ids_in = tf.keras.layers.Input(shape=(max_len,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(max_len,), name='masked_token', dtype='int32')

    input_ids_c1 = tf.keras.layers.Input(shape=(512,), name='input_token_c1', dtype='int32')
    input_masks_c1 = tf.keras.layers.Input(shape=(512,), name='masked_token_c1', dtype='int32')

    input_ids_c2 = tf.keras.layers.Input(shape=(512,), name='input_token_c2', dtype='int32')
    input_masks_c2 = tf.keras.layers.Input(shape=(512,), name='masked_token_c2', dtype='int32')

    embedding_layer_in = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
    embedding_layer_c1 = transformer_model(input_ids_c1, attention_mask=input_masks_c1)[0]
    embedding_layer_c2 = transformer_model(input_ids_c2, attention_mask=input_masks_c2)[0]
    embedding_layer = tf.keras.layers.concatenate([embedding_layer_in, embedding_layer_c1, embedding_layer_c2], axis=1)

    X = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=0.1))(embedding_layer)
    X = tf.keras.layers.GlobalMaxPool1D()(X)
    X = tf.keras.layers.Dense(hidden_size, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid')(X)
    model = tf.keras.Model(
        inputs=[input_ids_in, input_masks_in, input_ids_c1, input_masks_c1, input_ids_c2, input_masks_c2], outputs=X)

    for layer in model.layers[:8]:
        print(layer.name)
        layer.trainable = False

    return model

def BertModel_bilstm_context(max_len, hidden_size, learning_rate):
    config = BertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    transformer_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)

    input_ids_in = tf.keras.layers.Input(shape=(max_len,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(max_len,), name='masked_token', dtype='int32')

    input_ids_c1 = tf.keras.layers.Input(shape=(200,), name='input_token_c1', dtype='int32')
    input_masks_c1 = tf.keras.layers.Input(shape=(200,), name='masked_token_c1', dtype='int32')

    input_ids_c2 = tf.keras.layers.Input(shape=(200,), name='input_token_c2', dtype='int32')
    input_masks_c2 = tf.keras.layers.Input(shape=(200,), name='masked_token_c2', dtype='int32')

    embedding_layer_in = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
    embedding_layer_c1 = transformer_model(input_ids_c1, attention_mask=input_masks_c1)[0]
    embedding_layer_c2 = transformer_model(input_ids_c2, attention_mask=input_masks_c2)[0]
    embedding_layer = tf.keras.layers.concatenate([embedding_layer_in, embedding_layer_c1, embedding_layer_c2], axis=1)

    X = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding_layer)
    X = tf.keras.layers.GlobalMaxPool1D()(X)
    X = tf.keras.layers.Dense(hidden_size, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid')(X)
    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in, input_ids_c1, input_masks_c1, input_ids_c2, input_masks_c2], outputs=X)

    for layer in model.layers[:8]:
        print(layer.name)
        layer.trainable = False

    return model

def BertModel_bilstm_feedback(max_len, hidden_size, learning_rate, max_feedback_len):
    config = BertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    transformer_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)

    input_ids_in = tf.keras.layers.Input(shape=(max_len,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(max_len,), name='masked_token', dtype='int32')

    input_ids_fb = tf.keras.layers.Input(shape=(max_feedback_len,), name='input_token_fb', dtype='int32')
    input_masks_fb = tf.keras.layers.Input(shape=(max_feedback_len,), name='masked_token_fb', dtype='int32')

    embedding_layer_in = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
    embedding_layer_fb = transformer_model(input_ids_fb, attention_mask=input_masks_fb)[0]
    embedding_layer = tf.keras.layers.concatenate([embedding_layer_in, embedding_layer_fb], axis=1)

    X = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding_layer)
    X = tf.keras.layers.GlobalMaxPool1D()(X)
    X = tf.keras.layers.Dense(hidden_size, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid')(X)
    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in, input_ids_fb, input_masks_fb], outputs=X)

    for layer in model.layers[:6]:
        print(layer.name)
        layer.trainable = False

    return model

def BertModel_bilstm_context_feedback(max_len, hidden_size, learning_rate, max_context_len, max_feedback_len):
    config = BertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    transformer_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)

    input_ids_in = tf.keras.layers.Input(shape=(max_len,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(max_len,), name='masked_token', dtype='int32')

    input_ids_c1 = tf.keras.layers.Input(shape=(max_context_len,), name='input_token_c1', dtype='int32')
    input_masks_c1 = tf.keras.layers.Input(shape=(max_context_len,), name='masked_token_c1', dtype='int32')

    input_ids_c2 = tf.keras.layers.Input(shape=(max_context_len,), name='input_token_c2', dtype='int32')
    input_masks_c2 = tf.keras.layers.Input(shape=(max_context_len,), name='masked_token_c2', dtype='int32')

    input_ids_fb = tf.keras.layers.Input(shape=(max_feedback_len,), name='input_token_fb', dtype='int32')
    input_masks_fb = tf.keras.layers.Input(shape=(max_feedback_len,), name='masked_token_fb', dtype='int32')

    embedding_layer_in = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
    embedding_layer_c1 = transformer_model(input_ids_c1, attention_mask=input_masks_c1)[0]
    embedding_layer_c2 = transformer_model(input_ids_c2, attention_mask=input_masks_c2)[0]
    embedding_layer_fb = transformer_model(input_ids_fb, attention_mask=input_masks_fb)[0]
    embedding_layer = tf.keras.layers.concatenate([embedding_layer_in, embedding_layer_c1, embedding_layer_c2, embedding_layer_fb], axis=1)

    X = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding_layer)
    X = tf.keras.layers.GlobalMaxPool1D()(X)
    X = tf.keras.layers.Dense(hidden_size, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid')(X)

    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in, input_ids_c1, input_masks_c1, input_ids_c2, input_masks_c2, input_ids_fb, input_masks_fb], outputs=X)

    for layer in model.layers[:10]:
        print(layer.name)
        layer.trainable = False

    return model

def BertModel_bilstm_multitask(max_len, hidden_size, learning_rate):
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

    purpose_output = tf.keras.layers.Dense(1, activation='sigmoid')(X)
    desirability_output = tf.keras.layers.Dense(1, activation='sigmoid')(X)


    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=[purpose_output, desirability_output])

    for layer in model.layers[:3]:
        print(layer.name)
        layer.trainable = False

    return model

def BertModel_bilstm_context_handfeatures(max_len, hidden_size, learning_rate):
    config = BertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    transformer_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)

    input_ids_in = tf.keras.layers.Input(shape=(max_len,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(max_len,), name='masked_token', dtype='int32')
    features = tf.keras.layers.Input(shape=(10,), name='context_features', dtype='float32')

    embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
    X = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding_layer)
    X = tf.keras.layers.GlobalMaxPool1D()(X)

    merged = tf.keras.layers.concatenate([X,features])

    X = tf.keras.layers.Dense(hidden_size, activation='relu')(merged)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid')(X)
    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in, features], outputs=X)

    for layer in model.layers[:3]:
        print(layer.name)
        layer.trainable = False

    return model










