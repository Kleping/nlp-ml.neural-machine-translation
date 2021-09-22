import numpy as np
import tensorflow as tf
import random as rn
import json
import os

EPOCHS = 1
BATCH_SIZE = 32
LATENT_DIM = 128
VS = 25000
BATCH_SLICE = 64

NAME_MODEL = 'nmt'
NAME_TRAIN_DATA = 'encoded_data_train'
NAME_VALID_DATA = 'encoded_data_valid'

ROOT = ''
ROOT_DATA = 'data/'
ROOT_MODELS = 'models/'

PATH_LEARNING_STATE = '{}{}learning_state.config'.format(ROOT, ROOT_DATA)
PATH_TRAIN_DATA = '{}{}{}.txt'.format(ROOT, ROOT_DATA, NAME_TRAIN_DATA)
PATH_VALID_DATA = '{}{}{}.txt'.format(ROOT, ROOT_DATA, NAME_VALID_DATA)
PATH_TRAIN_CONFIG = '{}{}{}.config'.format(ROOT, ROOT_DATA, NAME_TRAIN_DATA)
PATH_VALID_CONFIG = '{}{}{}.config'.format(ROOT, ROOT_DATA, NAME_VALID_DATA)
PATH_MODEL = '{}{}{}'.format(ROOT, ROOT_MODELS, NAME_MODEL)

ID_BOS = 1
ID_EOS = 2


def init_learning_state():
    _len_train = len(open(PATH_TRAIN_DATA, encoding='utf-8').readlines())
    _len_valid = len(open(PATH_VALID_DATA, encoding='utf-8').readlines())

    _range_train = list(range(_len_train))
    _range_valid = list(range(_len_valid))

    rn.shuffle(_range_train)
    rn.shuffle(_range_valid)

    with open(PATH_LEARNING_STATE, 'w', encoding='utf-8') as _f:
        _data_learning_state = {
            'reserve_train': _range_train,
            'reserve_valid': _range_valid,
            'len_train': _len_train,
            'len_valid': _len_valid,
            'epochs': 0,
        }

        json.dump(_data_learning_state, _f, ensure_ascii=False, indent=4)

    return _data_learning_state


def read_learning_state():
    if os.path.isfile(PATH_LEARNING_STATE):
        _learning_state = json.load(open(PATH_LEARNING_STATE, encoding='utf-8'))
        _learning_state['reserve_valid'] = shuffle(list(range(_learning_state['len_valid'])))
        print('learning state loaded')
    else:
        _learning_state = init_learning_state()
        print('learning state created')

    return _learning_state


def write_learning_state(_data_learning_state):
    with open(PATH_LEARNING_STATE, 'w', encoding='utf-8') as _f:
        json.dump(_data_learning_state, _f, ensure_ascii=False, indent=4)
    print('learning state saved')


def shuffle(_data):
    rn.shuffle(_data)
    return _data


def save_model(_path, _model, _learning_state):
    write_learning_state(_learning_state)
    tf.keras.models.save_model(_model, '{}.h5'.format(_path))
    print('model saved')


def load_model(_path):

    if os.path.isfile('{}.h5'.format(_path)):
        _model = tf.keras.models.load_model('{}.h5'.format(_path))
        print('model loaded')
    else:
        _model = create_model()
        print('model created')

    _model = compile_model(_model)
    return _model


def encode(_x_collection, _y_collection, _batch_size, _max_x, _max_y):

    _encoded_x = np.zeros((_batch_size, _max_x), dtype='int32')
    _encoded_y = np.zeros((_batch_size, _max_y), dtype='int32')
    _decoded_y = np.zeros((_batch_size, _max_y), dtype='int32')

    # x encoding

    for _i_sample, _x_sample in enumerate(_x_collection):
        for _i_token, _x_id in enumerate(_x_sample.split(' ')):
            _encoded_x[_i_sample, _i_token] = int(_x_id)

    # DEBUGGING
    # print(_x_collection[0])
    # print(_encoded_x[0])
    # print()

    # y encoding

    for _i_sample, _y_sample in enumerate(_y_collection):
        for _i_token, _y_id in enumerate(_y_sample.split(' ')):
            _encoded_y[_i_sample, _i_token] = int(_y_id)

            if _i_token > 0:
                _decoded_y[_i_sample, _i_token-1] = int(_y_id)

    # DEBUGGING
    # print(_y_collection[0])
    # print(_encoded_y[0])
    # print(_decoded_y[0])
    # exit()

    return [_encoded_x, _encoded_y], _decoded_y


def encode_data_collection(_path_data, _batch, _batch_size, _max_x, _max_y):
    _x_collection = list()
    _y_collection = list()

    _max_batch_value = max(_batch)

    with open(_path_data, encoding='utf-8') as _f_data:
        _lines = _f_data.readlines()
        for _i in _batch:
            _x_sample, _y_sample = _lines[_i].split('\t')
            _x_collection.append(_x_sample)
            _y_collection.append('{} {} {}'.format(ID_BOS, _y_sample, ID_EOS))

    _x, _y = encode(_x_collection, _y_collection, _batch_size, _max_x, _max_y)
    return _x, _y


def read_valid_batch(_max_x, _max_y, _batch_size, _learning_state):
    _reserve = _learning_state['reserve_valid']
    _overridable_batch_size = _batch_size

    if len(_reserve) >= _overridable_batch_size:
        _batch = _reserve[:_overridable_batch_size]
        _learning_state['reserve_valid'] = _reserve[_overridable_batch_size:]
    elif len(_reserve) == 0:
        _learning_state['reserve_valid'] = shuffle(list(range(_learning_state['len_valid'])))
        return None, None, _learning_state
    else:
        _overridable_batch_size = len(_reserve)
        _batch = _reserve
        _learning_state['reserve_valid'] = []

    _x, _y = encode_data_collection(PATH_VALID_DATA, _batch, _overridable_batch_size, _max_x, _max_y)
    return _x, _y, _learning_state


def read_train_batch(_max_x, _max_y, _batch_size, _learning_state):
    _reserve = _learning_state['reserve_train']
    _overridable_batch_size = _batch_size

    if len(_reserve) >= _overridable_batch_size:
        _batch = _reserve[:_overridable_batch_size]
        _learning_state['reserve_train'] = _reserve[_overridable_batch_size:]
    elif len(_reserve) == 0:
        _range_data = shuffle(list(range(_learning_state['len_train'])))
        _batch = _range_data[:_overridable_batch_size]
        _learning_state['reserve_train'] = _range_data[_overridable_batch_size:]
        _learning_state['epochs'] += 1
    else:
        _overridable_batch_size = len(_reserve)
        _batch = _reserve
        _learning_state['reserve_train'] = []

    _x, _y = encode_data_collection(PATH_TRAIN_DATA, _batch, _overridable_batch_size, _max_x, _max_y)
    return _x, _y, _learning_state


def read_data_configs():
    _config_train = json.load(open(PATH_TRAIN_CONFIG, encoding='utf-8'))
    _config_valid = json.load(open(PATH_VALID_CONFIG, encoding='utf-8'))
    return _config_train['max_x'], _config_train['max_y'] + 2, _config_valid['max_x'], _config_valid['max_y'] + 2


def create_model():
    encoder_inputs = tf.keras.Input(shape=(None,))
    encoder_emb = tf.keras.layers.Embedding(VS, LATENT_DIM)(encoder_inputs)

    encoder_0 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            LATENT_DIM * 2,
            return_sequences=True,
            dropout=.4,
            recurrent_dropout=.4
        )
    )

    encoder_1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            LATENT_DIM * 2,
            return_sequences=True,
            dropout=.4,
            recurrent_dropout=.4
        )
    )

    encoder_2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            LATENT_DIM * 2,
            return_sequences=True,
            dropout=.4,
            recurrent_dropout=.4
        )
    )

    encoder_last = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            LATENT_DIM * 2,
            return_sequences=True,
            return_state=True,
            dropout=.4,
            recurrent_dropout=.4
        )
    )

    encoder_output_0 = encoder_0(encoder_emb)
    encoder_output_1 = encoder_1(encoder_output_0)
    encoder_output_2 = encoder_2(encoder_output_1)
    encoder_stack, forward_h, forward_c, backward_h, backward_c = encoder_last(encoder_output_2)

    encoder_last_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    encoder_last_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

    encoder_states = [encoder_last_h, encoder_last_c]

    decoder_inputs = tf.keras.Input(shape=(None,))
    decoder_emb = tf.keras.layers.Embedding(VS, LATENT_DIM)(decoder_inputs)

    decoder = tf.keras.layers.LSTM(
        LATENT_DIM * 4,
        return_sequences=True,
        return_state=True,
        dropout=.4,
        recurrent_dropout=.4
    )

    decoder_stack_h, _, _ = decoder(decoder_emb, initial_state=encoder_states)

    context = tf.keras.layers.Attention()([decoder_stack_h, encoder_stack])
    decoder_concat_input = tf.keras.layers.concatenate([context, decoder_stack_h])

    dense = tf.keras.layers.Dense(VS, activation='softmax')
    decoder_stack_h = tf.keras.layers.TimeDistributed(dense)(decoder_concat_input)

    return tf.keras.Model([encoder_inputs, decoder_inputs], decoder_stack_h)


def compile_model(m):
    _optimizer = tf.keras.optimizers.Adam()
    _loss = tf.keras.losses.SparseCategoricalCrossentropy()
    _metric = tf.keras.metrics.SparseCategoricalAccuracy()

    m.compile(optimizer=_optimizer, loss=_loss, metrics=[_metric])
    print('model compiled')
    return m


max_x_train, max_y_train, max_x_valid, max_y_valid = read_data_configs()
model = load_model(PATH_MODEL)
# print(model.summary())


learning_state = read_learning_state()

while learning_state < EPOCHS:

    if len(learning_state['reserve_train']) > 0:
        x_train, y_train, learning_state = read_train_batch(max_x_train, max_y_train, BATCH_SIZE, learning_state)


shifted_batch_counter = batch_counter_train = 0
batch_counter_valid = 0
loss_train = acc_train = loss_valid = acc_valid = .0
current_epoch = -1


while current_epoch < EPOCHS:
    x_train, y_train, learning_state = read_train_batch(max_x_train, max_y_train, BATCH_SIZE, learning_state)

    if current_epoch == -1:
        len_reserve_train = len(learning_state['reserve_train'])
        diff = learning_state['len_train'] - len_reserve_train
        if diff != 0:
            shifted_batch_counter = diff / BATCH_SIZE - 1

        current_epoch = learning_state['epochs']

    if current_epoch != learning_state['epochs']:
        x_valid, y_valid, learning_state = read_valid_batch(max_x_valid, max_y_valid, BATCH_SIZE, learning_state)
        batch_counter_valid += 1

        b = int(learning_state['len_valid'] / BATCH_SIZE) + (0 if learning_state['len_valid'] % BATCH_SIZE == 0 else 1)
        print('{}\t\t{}'.format(batch_counter_valid, b))

        while x_valid is not None and y_valid is not None:
            loss_valid, acc_valid = model.test_on_batch(x_valid, y_valid, reset_metrics=False)
            x_valid, y_valid, learning_state = read_valid_batch(max_x_valid, max_y_valid, BATCH_SIZE, learning_state)
            batch_counter_valid += 1
            print('{}\t\t{}'.format(batch_counter_valid, b))

        print('{} [{:.5f} {:.5f}]\t[{:.5f} {:.5f}]\n'
              .format(current_epoch, loss_train, acc_train, loss_valid, acc_valid))

        batch_counter_train = 0
        shifted_batch_counter = 0
        model.reset_metrics()

    loss_train, acc_train = model.train_on_batch(x_train, y_train, reset_metrics=False)

    if current_epoch != learning_state['epochs']:
        save_model(PATH_MODEL, model, learning_state)
        current_epoch = learning_state['epochs']

    batch_counter_train += 1

    a = int(batch_counter_train + shifted_batch_counter)
    b = int(learning_state['len_train'] / BATCH_SIZE) + (0 if learning_state['len_train'] % BATCH_SIZE == 0 else 1)
    if a % BATCH_SLICE == 0 or a == b:
        print('{} of {}\t{:.5f}\t{:.5f}'.format(a, b, loss_train, acc_train))
        save_model(PATH_MODEL, model, learning_state)
