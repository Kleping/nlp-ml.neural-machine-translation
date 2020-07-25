import tensorflow as tf

from classes.constant import MODEL_PATH, MAX_SEQUENCE
from classes.auxiliary import linear_regression_equality


def compile_model(model):
    model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=[linear_regression_equality])
    return model


def create_model(num_encoder_tokens=20, num_decoder_tokens=20, latent_dim=128):
    # Define an input sequence and process it.
    encoder_inputs = tf.keras.Input(shape=(None, num_encoder_tokens))
    encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = tf.keras.Input(shape=(None, num_decoder_tokens))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model = compile_model(model)
    print(model.summary())
    return model


def attention_block(decoder, input, encoder_output, encoder_state, dense, voc_size):
    decoder_output, _, _ = decoder(input, initial_state=encoder_state)
    context = tf.keras.layers.Attention()([encoder_output, decoder_output])
    decoder_combined_context = tf.keras.layers.Concatenate()([context, decoder_output])
    d0 = tf.keras.layers.Dense(dense, activation="relu")
    d1 = tf.keras.layers.Dense(voc_size, activation="softmax")
    in_dense = tf.keras.layers.TimeDistributed(d0)(decoder_combined_context)
    ou_dense = tf.keras.layers.TimeDistributed(d1)(in_dense)
    return ou_dense

def restore_model(n_units):
    model = compile_model(tf.keras.models.load_model(MODEL_PATH, compile=False))
    encoder_input = model.input[0]
    encoder_output, encoder_h, encoder_c = model.layers[3].output
    encoder_state = [encoder_h, encoder_c]
    encoder_model = tf.keras.Model([encoder_input], [encoder_output, encoder_state])
    print('encoder_model: ', encoder_input.shape)

    decoder_input = model.input[1]
    decoder_emb = model.layers[4].output
    attention_input = tf.keras.Input(shape=(None, n_units,), name="in_1")
    decoder = model.layers[5]
    initial_h = tf.keras.Input(shape=(n_units,), name="in_2")
    initial_c = tf.keras.Input(shape=(n_units,), name="in_3")
    decoder_initial_state = [initial_h, initial_c]

    decoder_output, decoder_h, decoder_c = decoder(decoder_emb, initial_state=decoder_initial_state)
    decoder_output_state = [decoder_h, decoder_c]

    context = model.layers[6]([attention_input, decoder_output])
    decoder_combined_context = model.layers[7]([context, decoder_output])
    attention_output = model.layers[8](decoder_combined_context)
    output = model.layers[9](attention_output)

    decoder_model = tf.keras.Model([decoder_input, attention_input, decoder_initial_state],
                                   [output] + decoder_output_state)

    print('decoder_model: ', decoder_input.shape, attention_input.shape, (initial_h.shape, initial_c.shape))

    return encoder_model, decoder_model
