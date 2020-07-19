import re
import numpy as np
import random

from classes.constant import MAX_SEQUENCE, SENTINELS, ACCEPTED_DIFF, BATCH_SIZE


def get_lines(path, formatted):
    lines = list()
    with open(path, "r", encoding='utf-8') as file:
        [lines.append(formatted(i)) for i in file.readlines()]
    return lines


def split_with_keep_delimiters(string, delimiters):
    return re.split('(' + '|'.join(map(re.escape, delimiters)) + ')', string)


def tokenize_sequence(seq):
    return seq.split()


def encode_seq(seq, voc):
    encoded_input = np.zeros((MAX_SEQUENCE, len(voc)), dtype='float32')
    for i in range(len(seq)):
        c = voc.index(seq[i])
        # a number of sample, an index of position in the current sentence,
        # an index of character in the vocabulary
        encoded_input[i, c] = 1.
    return encoded_input


def decompose_tokens(tokens, shuffle):
    decomposed = list()
    for i, token in enumerate(tokens):
        decomposed.append(tokens[:i+1])
    if shuffle:
        random.shuffle(decomposed)
    return decomposed


def clothe_to(str_list, symbols):
    str_list.insert(0, symbols[0])
    str_list.append(symbols[1])
    return str_list


def seq_to_tokens(seq, voc):
    return [voc[np.argmax(seq[i, :])] for i in range(len(seq))]


def decode_seq(input_seq, encoder_model, decoder_model, voc):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, len(voc)))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, voc.index(SENTINELS[0])] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = voc[sampled_token_index]
        decoded_sentence += sampled_token

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_token == SENTINELS[1] or len(decoded_sentence) > MAX_SEQUENCE:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, len(voc)))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


def linear_regression_equality(y_true, y_pred):
    import tensorflow as tf
    diff = tf.keras.backend.abs(y_true - y_pred)
    return tf.keras.backend.mean(tf.keras.backend.cast(diff < ACCEPTED_DIFF, 'float32'))


def get_voc(data):
    voc = SENTINELS
    delimiters = [' ']
    for k in data:
        [[voc.append(w) for w in split_with_keep_delimiters(s, delimiters) if w not in voc] for s in data[k]]
    voc = sorted(voc)
    return voc


def split_data(data, coefficient):
    validation = list()
    train = list()

    for k in data:
        cluster = data[k]
        cluster_len = int(len(cluster) * coefficient // len(data))
        [validation.append(i) for i in cluster[-cluster_len:]]
        [train.append(i) for i in cluster[:int(len(cluster) - cluster_len)]]

    random.shuffle(validation)
    random.shuffle(train)
    return train, validation


def calculate_steps(train, validation):
    steps_per_epoch = int(len(train) // BATCH_SIZE)
    validation_steps = int(len(validation) // BATCH_SIZE)
    return steps_per_epoch, validation_steps
