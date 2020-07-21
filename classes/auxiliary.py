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
    d_type = 'int32'
    arr = np.array([voc.index(seq[i]) for i in range(len(seq))], dtype=d_type)
    return np.append(arr, np.zeros(shape=MAX_SEQUENCE - len(seq), dtype=d_type))


def decompose_tokens(tokens, shuffle):
    decomposed = list()
    for i, token in enumerate(tokens):
        decomposed.append(tokens[:i+1])
    if shuffle:
        random.shuffle(decomposed)
    return decomposed


def clothe_to(str_list, symbols):
    new_list = list(str_list)
    new_list.insert(0, symbols[0])
    new_list.append(symbols[1])
    return new_list


def seq_to_tokens(seq, voc):
    return [voc[seq[i]] for i in range(len(seq))]


def decode_seq(input_seq, encoder_model, decoder_model, voc):
    encoder_output, encoder_state = encoder_model.predict(np.expand_dims(input_seq, axis=0))
    target_seq = np.zeros((1, 1, len(voc)))
    target_seq[0, 0, voc.index(SENTINELS[0])] = 1.

    stop_condition = False
    decoded_tokens = list()
    while not stop_condition:
        x = len(decoded_tokens) + 1
        output_tokens, h, c = decoder_model.predict([target_seq, encoder_output, encoder_state])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = voc[sampled_token_index]
        decoded_tokens.append(sampled_token)

        if sampled_token == SENTINELS[1] or len(decoded_tokens) > MAX_SEQUENCE:
            stop_condition = True

        # target_seq = np.zeros((1, 1, len(voc)))
        target_seq[0, 0, sampled_token_index] = 1.

        encoder_state = [h, c]

    return decoded_tokens


def linear_regression_equality(y_true, y_pred):
    import tensorflow as tf
    diff = tf.keras.backend.abs(y_true - y_pred)
    return tf.keras.backend.mean(tf.keras.backend.cast(diff < ACCEPTED_DIFF, 'float32'))


def get_voc(sequences):
    voc = SENTINELS
    delimiters = [' ']
    for k in sequences:
        [[voc.append(w) for w in split_with_keep_delimiters(s, delimiters) if w not in voc] for s in sequences[k]]
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
