import numpy as np
import tensorflow as tf
import json
from bpemb import BPEmb

UNITS = 512

NAME_TRAIN_DATA = 'encoded_data_train'
NAME_VALID_DATA = 'encoded_data_valid'

NAME_MODEL = 'nmt'
model_dir = 'models'
model_path = '{}/{}.h5'.format(model_dir, NAME_MODEL)

ROOT = ''
ROOT_DATA = 'data/'

PATH_INFERENCE_DATA = 'data/inference.txt'
PATH_TRAIN_CONFIG = '{}{}{}.config'.format(ROOT, ROOT_DATA, NAME_TRAIN_DATA)
PATH_VALID_CONFIG = '{}{}{}.config'.format(ROOT, ROOT_DATA, NAME_VALID_DATA)

VS = 25000
BPE_RU = BPEmb(lang='ru', vs=VS, dim=50)
BPE_EN = BPEmb(lang='en', vs=VS, dim=50)
ID_BOS = 1
ID_EOS = 2


def read_data_configs():
    _config_train = json.load(open(PATH_TRAIN_CONFIG, encoding='utf-8'))
    _config_valid = json.load(open(PATH_VALID_CONFIG, encoding='utf-8'))
    return _config_train['max_x'], _config_train['max_y'] + 2, _config_valid['max_x'], _config_valid['max_y'] + 2


model = tf.keras.models.load_model('{}/{}.h5'.format(model_dir, NAME_MODEL))
# print(model.summary())
layers = dict((i.name, i) for i in model.layers)

encoder_input = model.inputs[0]
encoder_output, forward_last_h, forward_last_c, backward_last_h, backward_last_c = layers['bidirectional_2'].output

encoder_last_h = tf.keras.layers.Concatenate()([forward_last_h, backward_last_h])
encoder_last_c = tf.keras.layers.Concatenate()([forward_last_c, backward_last_c])

encoder_model = tf.keras.Model(encoder_input, [encoder_output] + [encoder_last_h, encoder_last_c])

decoder_input = model.inputs[1]
decoder_state_input_h = tf.keras.Input(shape=(UNITS*2,), name="custom_input_3")
decoder_state_input_c = tf.keras.Input(shape=(UNITS*2,), name="custom_input_4")
encoder_stack_h = tf.keras.Input(shape=(None, UNITS*2,), name="custom_input_5")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder = layers['decoder']
decoder_emb = layers['decoder_embedding'](decoder_input)
decoder_stack_h, decoder_last_h, decoder_last_c = decoder(decoder_emb, initial_state=decoder_states_inputs)

decoder_attention = layers['attention']
decoder_concatenate = layers['concatenate']
decoder_dense = layers['time_distributed_1']

decoder_last = [decoder_last_h, decoder_last_c]
context = decoder_attention([decoder_stack_h, encoder_stack_h])

decoder_stack_h = decoder_dense(tf.keras.layers.concatenate([context, decoder_stack_h]))
decoder_model = tf.keras.Model(
    [decoder_input, encoder_stack_h, decoder_states_inputs],
    [decoder_stack_h] + decoder_last
)


with open(PATH_INFERENCE_DATA, 'r', encoding='utf-8') as f:
    data = f.read().split('\n')

max_x_train, max_y_train, _, _ = read_data_configs()


def encode_input(_text):
    _encoded_input = np.zeros((1, max_x_train), dtype='int32')

    for _i, _token in enumerate(BPE_RU.encode_ids(_text)):
        _encoded_input[0, _i] = _token

    return _encoded_input


def predict(_input):
    encoder_outputs, h, c = encoder_model.predict(_input)
    states_value = [h, c]
    target_text = np.zeros((1, 1), dtype='int32')
    target_text[0, 0] = ID_BOS

    stop_condition = False
    predicted_tokens = list()

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_text, encoder_outputs, states_value])
        sampled_token = np.argmax(output_tokens[0, -1, :])
        predicted_tokens.append(int(sampled_token))
        if sampled_token == ID_EOS or len(predicted_tokens) > max_y_train:
            stop_condition = True

        target_text = np.zeros((1, 1), dtype='int32')
        target_text[0, 0] = sampled_token
        states_value = [h, c]

    return list(filter(lambda a: a != 0, predicted_tokens))


for i in range(len(data)):
    source_text = data[i]
    formatted_result = ''
    result = BPE_EN.decode_ids(predict(encode_input(source_text)))
    alpha_checked = False
    source_digits = [char for char in source_text if char.isdigit()]

    for i_char, char in enumerate(result):
        if char.isalpha() and alpha_checked is False:
            result = result[:i_char] + char.upper() + result[i_char+1:]
            alpha_checked = True
        elif char.isdigit() and len(source_digits) > 0:
            result = result[:i_char] + source_digits.pop(0) + result[i_char + 1:]

    print(result)
