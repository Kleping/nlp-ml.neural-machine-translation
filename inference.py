from tensorflow.keras.preprocessing.text import tokenizer_from_json, text_to_word_sequence
import numpy as np
import tensorflow as tf
import stanza
import spacy_stanza
import json

stanza.download('ru')
nlp = spacy_stanza.load_pipeline('ru')

LATENT_DIM = 128

ROOT = ''
ROOT_DATA = 'data/'
FILTERS = r"-<>\[\]()#•@—№*“…”–%+’&/»«.\",!?:;"
LOWER = False
SPLIT = ' '

model_name = 'ie'
model_dir = 'models'
model_path = '{}/{}.h5'.format(model_dir, model_name)
PATH_INFERENCE_DATA = 'data/inference.txt'

PATH_TOKENIZER_CONFIG_TRAIN = '{}{}tokenizer_train_config.json'.format(ROOT, ROOT_DATA)

TOKEN_BEG = 'BOS'
TOKEN_END = 'EOS'
TOKEN_OOV = 'OOV'
TOKEN_PAD = 'PAD'
TOKEN_ORG = 'ORG'
TOKEN_PER = 'PER'

VOC_TARGET = [TOKEN_PAD, TOKEN_OOV, TOKEN_PER, TOKEN_ORG, TOKEN_BEG, TOKEN_END]


def _load_tokenizers_from_json():
    with open(PATH_TOKENIZER_CONFIG_TRAIN) as _f:
        _tokenizer_train = tokenizer_from_json(json.load(_f))

    return _tokenizer_train


def _read_config():
    with open('{}/{}.config'.format(model_dir, model_name), 'r') as config_file:
        config = json.load(config_file)

    return config['max_source'], config['max_target']


with open(PATH_INFERENCE_DATA, 'r', encoding='utf-8') as f:
    data = f.read().split('\n')

MAX_SOURCE, MAX_TARGET = _read_config()
tokenizer_train = _load_tokenizers_from_json()
NUM_WORDS_TRAIN = len(tokenizer_train.word_counts) + 2

model = tf.keras.models.load_model('{}/{}.h5'.format(model_dir, model_name))
print(model.summary())
encoder_input = model.inputs[0]
encoder_output, forward_last_h, forward_last_c, backward_last_h, backward_last_c = model.layers[6].output

encoder_last_h = tf.keras.layers.Concatenate()([forward_last_h, backward_last_h])
encoder_last_c = tf.keras.layers.Concatenate()([forward_last_c, backward_last_c])

encoder_model = tf.keras.Model(encoder_input, [encoder_output] + [encoder_last_h, encoder_last_c])

decoder_input = model.inputs[1]
decoder_state_input_h = tf.keras.Input(shape=(LATENT_DIM*4,), name="input_3")
decoder_state_input_c = tf.keras.Input(shape=(LATENT_DIM*4,), name="input_4")
encoder_stack_h = tf.keras.Input(shape=(None, LATENT_DIM*4,), name="input_5")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder = model.layers[10]
decoder_emb = model.layers[7](decoder_input)
decoder_stack_h, decoder_last_h, decoder_last_c = decoder(decoder_emb, initial_state=decoder_states_inputs)

decoder_attention = model.layers[11]
decoder_concatenate = model.layers[12]
decoder_dense = model.layers[13]

decoder_last = [decoder_last_h, decoder_last_c]
context = decoder_attention([decoder_stack_h, encoder_stack_h])

decoder_stack_h = decoder_dense(tf.keras.layers.concatenate([context, decoder_stack_h]))
decoder_model = tf.keras.Model(
    [decoder_input, encoder_stack_h, decoder_states_inputs],
    [decoder_stack_h] + decoder_last
)


def _lemmatize_text(_text):
    _lemmatized_text = _text
    _doc = nlp(_text)
    for _token in _doc:
        _lemmatized_text = _lemmatized_text.replace(str(_token), _token.lemma_)
    return _lemmatized_text


def _text_to_word_sequence(_sentence):
    return text_to_word_sequence(_sentence, lower=LOWER, filters=FILTERS)


def _encode_input(_text):
    _text = _lemmatize_text(_text)
    _encoded_input = np.zeros((1, MAX_SOURCE), dtype='int32')
    _sequence = tokenizer_train.texts_to_sequences([_text])[0]
    for _idx, _token in enumerate(_sequence):
        _encoded_input[0, _idx] = _token

    return _encoded_input


def _predict(input_sample):
    encoder_outputs, h, c = encoder_model.predict(input_sample)
    states_value = [h, c]
    target_text = np.zeros((1, 1), dtype='int32')
    target_text[0, 0] = VOC_TARGET.index(TOKEN_BEG)

    stop_condition = False
    predicted_tokens = list()

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_text, encoder_outputs, states_value])
        sampled_token = np.argmax(output_tokens[0, -1, :])
        predicted_tokens.append(sampled_token)
        if sampled_token == VOC_TARGET.index(TOKEN_END) or len(predicted_tokens) > MAX_TARGET:
            stop_condition = True

        target_text = np.zeros((1, 1), dtype='int32')
        target_text[0, 0] = sampled_token
        states_value = [h, c]
    return predicted_tokens[:-1]


for i in range(len(data)):
    source_text = data[i].split('\t')[0]
    formatted_result = ''
    result = _predict(_encode_input(source_text))
    w_seq = _text_to_word_sequence(source_text)

    for _idx, _i in enumerate(result):
        if VOC_TARGET[int(_i)] is TOKEN_PER or VOC_TARGET[int(_i)] is TOKEN_ORG:
            w = w_seq[_idx]
            formatted_result += '{} '.format(str(source_text.index(w)))
            formatted_result += '{} '.format(len(w))
            formatted_result += '{} '.format('PERSON' if VOC_TARGET[int(_i)] is TOKEN_PER else TOKEN_ORG)

    formatted_result += 'EOL'
    # predicted_text = '\n'.join([
    #     '{}: {}'.format(VOC_TARGET[int(i)], w_seq[idx])
    #     for idx, i in enumerate(result)
    #     if VOC_TARGET[int(i)] is TOKEN_PER or VOC_TARGET[int(i)] is TOKEN_ORG
    # ])

    print('{}'.format(formatted_result))
