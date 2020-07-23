import tensorflow as tf

from classes.constant import MODEL_PATH, MAX_SEQUENCE
from classes.auxiliary import linear_regression_equality


def compile_model(model):
    model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=[linear_regression_equality])
    return model


def create_model(n_voc, n_dim=128, n_emb_dim=128):
    encoder_input = tf.keras.layers.Input(shape=(MAX_SEQUENCE,))
    encoder_embedding = tf.keras.layers.Embedding(n_voc, n_emb_dim, input_length=MAX_SEQUENCE)(encoder_input)
    encoder = tf.keras.layers.LSTM(n_dim, return_sequences=True, return_state=True)
    encoder_output, state_h, state_c = encoder(encoder_embedding)
    encoder_state = [state_h, state_c]

    decoder_input = tf.keras.layers.Input(shape=(MAX_SEQUENCE,))
    decoder_embedding = tf.keras.layers.Embedding(n_voc, n_emb_dim, input_length=MAX_SEQUENCE)(decoder_input)
    decoder = tf.keras.layers.LSTM(n_dim, return_sequences=True, return_state=True)

    # seq2seq
    # decoder_output, _, _ = decoder(decoder_embedding, initial_state=encoder_state)
    # decoder_dense = tf.keras.layers.Dense(n_voc, activation="softmax")
    # output = decoder_dense(decoder_output)

    # seq2seq with attention
    decoder_output, _, _ = decoder(decoder_embedding, initial_state=encoder_state)
    context = tf.keras.layers.Attention()([encoder_output, decoder_output])
    decoder_combined_context = tf.keras.layers.Concatenate()([context, decoder_output])
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_dim, activation="relu"))(decoder_combined_context)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_voc, activation="softmax"))(output)

    model = tf.keras.Model([encoder_input, decoder_input], output)
    model = compile_model(model)
    print(model.summary())
    return model


def restore_model(n_units):
    model = compile_model(tf.keras.models.load_model(MODEL_PATH, compile=False))

    encoder_input = model.input[0]
    encoder_output, encoder_h, encoder_c = model.layers[1].output
    encoder_state = [encoder_h, encoder_c]
    encoder_model = tf.keras.Model([encoder_input], [encoder_output, encoder_state])

    decoder_input = model.input[1]
    attention_input = tf.keras.Input(shape=(None, n_units,), name="in_1")
    decoder = model.layers[3]
    decoder_initial_h = tf.keras.Input(shape=(n_units,), name="in_2")
    decoder_initial_c = tf.keras.Input(shape=(n_units,), name="in_3")
    decoder_initial_state = [decoder_initial_h, decoder_initial_c]

    decoder_output, decoder_h, decoder_c = decoder(decoder_input, initial_state=decoder_initial_state)
    decoder_output_state = [decoder_h, decoder_c]

    context = model.layers[4]([attention_input, decoder_output])
    decoder_combined_context = model.layers[5]([context, decoder_output])
    attention_output = model.layers[6](decoder_combined_context)
    output = model.layers[7](attention_output)

    decoder_model = tf.keras.Model([decoder_input, attention_input, decoder_initial_state],
                                   [output] + decoder_output_state)

    return encoder_model, decoder_model
