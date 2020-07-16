import random
from classes import constant
from classes.DataSupplier import DataSupplier
from classes.auxiliary import get_lines, linear_regression_equality, get_vocabulary, \
                              split_data, calculate_steps, seq_to_text


def get_data(count_coefficient, shuffle, split_coefficient):
    data = dict()
    for suffix in constant.DATA_SUFFIXES:
        normalized_data = get_lines('data/normalized/eng_' + suffix + '.txt', lambda l: l[:-1])
        incomplete_data = get_lines('data/incomplete/eng_' + suffix + '.txt', lambda l: l[:-1])

        normalized_data = normalized_data[:int(len(normalized_data) * count_coefficient)]
        incomplete_data = incomplete_data[:int(len(incomplete_data) * count_coefficient)]
        data[suffix] = normalized_data + incomplete_data

    if shuffle is True:
        [random.shuffle(data[k]) for k in data]

    voc, voc_size = get_vocabulary(data)
    # print(max([max([len(sentence.split()) for sentence in data[k]]) for k in data]))

    train, validation = split_data(data, split_coefficient)
    validation_generator = DataSupplier(constant.BATCH_SIZE, validation, voc, voc_size)
    generator = DataSupplier(constant.BATCH_SIZE, train, voc, voc_size)
    print('Data', str(len(train)) + '/' + str(len(validation)), voc_size)
    return (generator, validation_generator), calculate_steps(train, validation), (voc, voc_size)


def compile_model(model):
    model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=[linear_regression_equality])
    return model


def create_model(n_input, n_units):
    import tensorflow as tf
    encoder_input = tf.keras.layers.Input(shape=(None, n_input,))
    encoder = tf.keras.layers.LSTM(n_units, return_sequences=True, return_state=True)
    encoder_output, state_h, state_c = encoder(encoder_input)
    encoder_states = [state_h, state_c]

    decoder_input = tf.keras.layers.Input(shape=(None, n_input,))
    decoder = tf.keras.layers.LSTM(n_units, return_sequences=True)
    decoder_output = decoder(decoder_input, initial_state=encoder_states)

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


def restore_model():
    import keras
    model = keras.models.load_model(constant.MODEL_PATH, compile=False)
    return compile_model(model)


def print_model_predictions():
    (train_data, validation_data), (_, _), (voc, voc_size) = get_data(.15, True, .2)
    model = restore_model()
    for i in range(20):
        input_data = validation_data.__getitem__(0)[0]
        encoded_input = input_data[0]
        decoded_input = input_data[1]
        input_seq = [encoded_input[i: i + 1], decoded_input[i: i + 1]]
        print('input ' + seq_to_text(input_seq[1], voc))
        output = seq_to_text(model.predict(input_seq), voc)[1:]

        for i in range(len(output)):
            if output[i] is constant.SENTINELS[1]:
                print('output ' + output[:i] + '\n')
                break


def train_model():
    (train_data, validation_data), (steps_per_epoch, validation_steps), (voc, voc_size) = get_data(.02, True, .2)
    model = create_model(voc_size, constant.LATENT_DIMENSIONS)

    model.fit_generator(generator=train_data,
                        validation_data=validation_data,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        epochs=constant.EPOCHS,
                        verbose=2,
                        use_multiprocessing=False,
                        shuffle=True)

    model.save(constant.MODEL_PATH)


def get_concrete_function():
    import tensorflow as tf

    model = tf.keras.models.load_model(constant.MODEL_PATH, compile=False)
    compile_model(model)

    full_model = tf.function(lambda x: model(x))

    x_tensor_spec = tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    y_tensor_spec = tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype)

    return full_model.get_concrete_function(x=[x_tensor_spec, y_tensor_spec])


def convert_to_tf_lite(path):
    import tensorflow as tf
    cf = get_concrete_function()

    converter = tf.lite.TFLiteConverter.from_concrete_functions([cf])
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(path, 'wb') as f:
        f.write(tflite_model)


def write_graph():
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    full_model = get_concrete_function()
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./models",
                      name=constant.MODEL_NAME + '.pb',
                      as_text=False)


def read_graph():
    import tensorflow as tf
    with tf.io.gfile.GFile('./models/' + constant.MODEL_NAME + '.pb', "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())
    return loaded

# write_graph()
# graph = read_graph()
# train_model()


convert_to_tf_lite('models/model.tflite')
