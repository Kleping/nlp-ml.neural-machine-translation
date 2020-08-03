#%%

import numpy as np
import tensorflow as tf
import random as rn
import sentencepiece as spm

#%%

language_tag = 'en'
latent_dim = 128
data_path = 'data/' + language_tag + '/encoded.txt'
model_prefix = 'models/' + language_tag + '/'
model_source_file = model_prefix + 'source.model'
model_target_file = model_prefix + 'target.model'
model_name = 'nmt'
max_source_seq_len = 13
max_target_seq_len = 16
n_test_samples = 20
voc_size_source = 100
voc_size_target = 102
i_bos = voc_size_source
i_eos = voc_size_source + 1

#%%

voc = dict()
with open('models/' + language_tag + '/target.vocab', "r", encoding="utf-8") as f:
    pairs = f.read().split("\n")
    for pair in pairs:
        if pair == '':
            continue
        v, k = pair.split('\t')
        if k == '0':
            continue
        voc[int(k[1:])] = v

#%%

data = list()
with open('data/' + language_tag + '/encoded.txt', "r", encoding='utf-8') as f:
    for sample in f.read().split("\n"):
        source, target = sample.split("\t")
        source_tokens = source.split()
        target_tokens = target.split()
        if (len(source_tokens) > max_source_seq_len) or (len(target_tokens) > max_target_seq_len):
            continue

        data.append(source)

#%%

model = tf.keras.models.load_model("models/" + model_name + ".h5")
encoder_input = model.inputs[0]
encoder_output, forward_last_h, forward_last_c, backward_last_h, backward_last_c = model.layers[1].output

encoder_last_h = tf.keras.layers.Concatenate()([forward_last_h, backward_last_h])
encoder_last_c = tf.keras.layers.Concatenate()([forward_last_c, backward_last_c])

encoder_model = tf.keras.Model(encoder_input, [encoder_output] + [encoder_last_h, encoder_last_c])

decoder_input = model.inputs[1]
decoder_state_input_h = tf.keras.Input(shape=(latent_dim*2,), name="input_3")
decoder_state_input_c = tf.keras.Input(shape=(latent_dim*2,), name="input_4")
encoder_stack_h = tf.keras.Input(shape=(None, latent_dim*2,), name="input_5")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder = model.layers[5]
decoder_stack_h, decoder_last_h, decoder_last_c = decoder(
    decoder_input, initial_state=decoder_states_inputs
)
decoder_attention = model.layers[6]
decoder_concatenate = model.layers[7]
decoder_dense = model.layers[8]

decoder_last = [decoder_last_h, decoder_last_c]
context = decoder_attention([decoder_stack_h, encoder_stack_h])

decoder_stack_h = decoder_dense(tf.keras.layers.concatenate([context, decoder_stack_h]))
decoder_model = tf.keras.Model(
    [decoder_input, encoder_stack_h, decoder_states_inputs],
    [decoder_stack_h] + decoder_last
)

#%%


def encode(sample):
    encoded_sample = np.zeros(shape=(1, max_source_seq_len, voc_size_source), dtype='float32')
    for t, i_vocab in enumerate(sample.split()):
        encoded_sample[0, t, int(i_vocab)] = 1.
    return encoded_sample


def predict(input_sample):
    encoder_outputs, h, c = encoder_model.predict(input_sample)
    states_value = [h, c]
    target_seq = np.zeros((1, 1, voc_size_target))
    target_seq[0, 0, i_bos] = 1.0

    stop_condition = False
    predicted_tokens = list()

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, encoder_outputs, states_value])

        sampled_token = np.argmax(output_tokens[0, -1, :])
        predicted_tokens.append(sampled_token)

        if sampled_token == i_eos or len(predicted_tokens) > max_target_seq_len:
            stop_condition = True

        target_seq = np.zeros((1, 1, voc_size_target))
        target_seq[0, 0, sampled_token] = 1.0

        states_value = [h, c]
    return predicted_tokens[:-1]


def decode(encoded_text, spp):
    return spp.Decode(encoded_text)


def encode_raw(raw_text, spp):
    return spp.Encode(raw_text, out_type=int, enable_sampling=False, alpha=.1, nbest_size=-1)


spp_source = spm.SentencePieceProcessor()
spp_target = spm.SentencePieceProcessor()
spp_source . Init(model_file=model_source_file)
spp_target . Init(model_file=model_target_file)


for i in rn.choices(range(len(data)), k=n_test_samples):
    sample = data[i]
    predicted_tokens = predict(encode(sample))
    print('\n')
    print('input:', decode([int(i) for i in sample.split()], spp_source))
    print('predicted:', decode([int(i) for i in predicted_tokens], spp_target))
