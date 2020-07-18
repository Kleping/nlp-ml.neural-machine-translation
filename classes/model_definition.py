import tensorflow as tf

from classes.constant import MODEL_PATH
from classes.auxiliary import linear_regression_equality


def compile_model(model):
    model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=[linear_regression_equality])
    return model


def create_model(n_input, n_units):
    encoder_input = tf.keras.layers.Input(shape=(None, n_input,))
    encoder = tf.keras.layers.LSTM(n_units, return_sequences=True, return_state=True)
    encoder_output, encoder_state_h, encoder_state_c = encoder(encoder_input)
    encoder_state = [encoder_state_h, encoder_state_c]

    decoder_input = tf.keras.layers.Input(shape=(None, n_input,))
    decoder = tf.keras.layers.LSTM(n_units, return_sequences=True, return_state=True)
    decoder_output, decoder_state_h, decoder_state_c = decoder(decoder_input, initial_state=encoder_state)

    # seq2seq
    # decoder_dense = tf.keras.layers.Dense(n_input, activation="softmax")
    # output = decoder_dense(decoder_output)

    #seq2seq with attention
    context = tf.keras.layers.Attention()([encoder_output, decoder_output])
    decoder_combined_context = tf.keras.layers.concatenate([context, decoder_output])
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_units, activation="relu"))(decoder_combined_context)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_input, activation="softmax"))(output)

    model = tf.keras.Model([encoder_input, decoder_input], output)
    model = compile_model(model)
    return model


def restore_model(n_units):
    model = compile_model(tf.keras.models.load_model(MODEL_PATH, compile=False))

    encoder_input = model.input[0]
    encoder_output, encoder_h, encoder_c = model.layers[1].output
    encoder_state = [encoder_h, encoder_c]
    encoder_model = tf.keras.Model(encoder_input, [encoder_state, encoder_output])

    decoder_input = model.input[1]
    decoder_input_attention = tf.keras.Input(shape=(None, n_units,))
    decoder = model.layers[3]
    decoder_initial_h = tf.keras.Input(shape=(n_units,), name='input_3')
    decoder_initial_c = tf.keras.Input(shape=(n_units,), name='input_4')
    decoder_initial_state = [decoder_initial_h, decoder_initial_c]

    decoder_output, decoder_h, decoder_c = decoder(decoder_input, initial_state=decoder_initial_state)
    decoder_output_state = [decoder_h, decoder_c]

    context = model.layers[4]([decoder_input_attention, decoder_output])
    decoder_combined_context = model.layers[5]([context, decoder_output])
    output = model.layers[6](decoder_combined_context)
    output = model.layers[7](output)

    decoder_model = tf.keras.Model([decoder_input, decoder_input_attention] + decoder_initial_state,
                                   [output] + decoder_output_state)

    return encoder_model, decoder_model
