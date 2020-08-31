import re
import numpy as np
import tensorflow as tf
import json

epochs = 100
batch_size = 128
coefficient = 100

num_data = coefficient*batch_size
latent_dim = 128
language_tag = 'en'
data_path = 'data/{}/paired.txt'.format(language_tag)
model_name = 'nmt_{}_{}_{}'.format(epochs, batch_size, coefficient)
validation_split = .2
punctuation = ['!', '?', '.', ',']
sentinels = ['<BOS>', '<EOS>']
OOV_TOKEN = '<OOV>'
PAD_TOKEN = '<PAD>'


def split_with_keep_delimiters(text, delimiters):
    return [
        token for token in re.split('(' + '|'.join(map(re.escape, delimiters)) + ')', text) if token is not ''
    ]


def tokenize_text(text):
    tokens = list()
    for token in text.split():
        if any(map(str.isdigit, token)):
            if token[-1] in punctuation:
                tokens.append(token[:-1])
                tokens.append(token[-1])
            else:
                tokens.append(token)
        else:
            [tokens.append(splitted_token) for splitted_token in split_with_keep_delimiters(token, punctuation)]

    return tokens


def encode_target(text):
    return '{} {} {}'.format(sentinels[0], text, sentinels[1])


def deserialize_and_read_config():
    with open('models/{}/{}.config'.format(language_tag, model_name), 'r') as config_file:
        config = json.load(config_file)

    return config['source'], config['target'], config['max_source_len'], config['max_target_len']


with open(data_path, 'r', encoding='utf-8') as f:
    data = f.read().split('\n')[:num_data]

source_voc, target_voc, max_source_seq_len, max_target_seq_len = deserialize_and_read_config()

model = tf.keras.models.load_model('models/{}/{}.h5'.format(language_tag, model_name))
print(model.summary())

encoder_input = model.inputs[0]
encoder_output, forward_last_h, forward_last_c, backward_last_h, backward_last_c = model.layers[3].output

encoder_last_h = tf.keras.layers.Concatenate()([forward_last_h, backward_last_h])
encoder_last_c = tf.keras.layers.Concatenate()([forward_last_c, backward_last_c])

encoder_model = tf.keras.Model(encoder_input, [encoder_output] + [encoder_last_h, encoder_last_c])

decoder_input = model.inputs[1]
decoder_state_input_h = tf.keras.Input(shape=(latent_dim*2,), name="input_3")
decoder_state_input_c = tf.keras.Input(shape=(latent_dim*2,), name="input_4")
encoder_stack_h = tf.keras.Input(shape=(None, latent_dim*2,), name="input_5")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder = model.layers[7]
decoder_stack_h, decoder_last_h, decoder_last_c = decoder(
    model.layers[4](decoder_input), initial_state=decoder_states_inputs
)
decoder_attention = model.layers[8]
decoder_concatenate = model.layers[9]
decoder_dense = model.layers[10]

decoder_last = [decoder_last_h, decoder_last_c]
context = decoder_attention([decoder_stack_h, encoder_stack_h])

decoder_stack_h = decoder_dense(tf.keras.layers.concatenate([context, decoder_stack_h]))
decoder_model = tf.keras.Model(
    [decoder_input, encoder_stack_h, decoder_states_inputs],
    [decoder_stack_h] + decoder_last
)


def get_token_index(voc, token):
    if token in voc:
        voc_i = voc.index(token)
    else:
        voc_i = voc.index(OOV_TOKEN)
    return voc_i


def encode(text):
    encoded_text = np.zeros((1, max_source_seq_len), dtype='int32')
    for t, token in enumerate(tokenize_text(text)):
        encoded_text[0, t] = get_token_index(source_voc, token)
    return encoded_text


def predict(input_sample):
    encoder_outputs, h, c = encoder_model.predict(input_sample)
    states_value = [h, c]
    target_text = np.zeros((1, max_target_seq_len), dtype='int32')
    target_text[0] = get_token_index(target_voc, sentinels[0])

    stop_condition = False
    predicted_tokens = list()

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_text, encoder_outputs, states_value])
        sampled_token = np.argmax(output_tokens[0, -1, :])
        predicted_tokens.append(sampled_token)

        if sampled_token == get_token_index(target_voc, sentinels[1]) or len(predicted_tokens) > max_target_seq_len:
            stop_condition = True
        target_text[0] = sampled_token
        states_value = [h, c]
    return predicted_tokens[:-1]


for i in range(len(data)):
    source_text, target_text = data[i].split('\t')
    encoded = encode(source_text)
    result = predict(encoded)
    predicted_text = ' '.join([target_voc[i] for i in result])
    print('{} {}'.format(source_text, predicted_text))
