import re
import numpy as np
from classes import constant
import keras as k
import random


def get_lines(path, formatted):
    lines = list()
    with open(path, "r", encoding='utf-8') as file:
        [lines.append(formatted(i)) for i in file.readlines()]
    return lines


def split_with_keep_delimiters(string, delimiters):
    return re.split('(' + '|'.join(map(re.escape, delimiters)) + ')', string)


def seq_to_text(encoded_input, voc):
    return ''.join([voc[np.argmax(encoded_input[0, i, :])] for i in range(len(encoded_input[0]))])


def punctuation_translate(x):
    return x.translate({ord(i): '' for i in constant.VOCABULARY_PUNCTUATION})


def find_max_sequence(samples):
    return max([len(sample) for sample in samples])


def decode_sequence(input_seq, encoder_model, decoder_model, vocabulary, vocabulary_len):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, vocabulary_len))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, vocabulary.index(constant.SENTINELS[0])] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = vocabulary[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == constant.SENTINELS[1] or len(decoded_sentence) > constant.MAX_SEQUENCE:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, vocabulary_len))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


def linear_regression_equality(y_true, y_pred):
    diff = k.backend.abs(y_true - y_pred)
    return k.backend.mean(k.backend.cast(diff < constant.ACCEPTED_DIFF, 'float32'))


def get_vocabulary(data):
    voc = constant.SENTINELS
    delimiters = [' ']
    for k in data:
        [[voc.append(w) for w in split_with_keep_delimiters(s, delimiters) if w not in voc] for s in data[k]]
    voc = sorted(voc)
    voc_size = len(voc)
    return voc, voc_size


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
    steps_per_epoch = int(len(train) // constant.BATCH_SIZE)
    validation_steps = int(len(validation) // constant.BATCH_SIZE)
    return steps_per_epoch, validation_steps
