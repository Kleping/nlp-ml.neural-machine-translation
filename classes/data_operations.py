import random

from classes.constant import DATA_SUFFIXES, SENTINELS, SHIFTED_SEQ_COUNT, BATCH_SIZE, MAX_SEQUENCE

from classes.DataSupplier import DataSupplier
from classes.auxiliary import get_lines, get_voc, split_data, calculate_steps, tokenize_sequence


def get_raw_data(count_coefficient, shuffle, assign_max_sequence=False):
    global MAX_SEQUENCE
    raw_data = dict()
    for suffix in DATA_SUFFIXES:
        normalized_data = get_lines('data/normalized/eng_' + suffix + '.txt', lambda l: l[:-1])
        incomplete_data = get_lines('data/incomplete/eng_' + suffix + '.txt', lambda l: l[:-1])

        if assign_max_sequence:
            concatenated_data = normalized_data + incomplete_data
            max_seq = max([len(tokenize_sequence(seq)) for seq in concatenated_data])
            if max_seq > MAX_SEQUENCE:
                MAX_SEQUENCE = max_seq

        if shuffle is True:
            random.shuffle(normalized_data)
            random.shuffle(incomplete_data)

        normalized_data = normalized_data[:int(len(normalized_data) * count_coefficient)]
        incomplete_data = incomplete_data[:int(len(incomplete_data) * count_coefficient)]
        raw_data[suffix] = normalized_data + incomplete_data

    if assign_max_sequence:
        MAX_SEQUENCE += len(SENTINELS) + SHIFTED_SEQ_COUNT
        print('assigned_max_sequence(' + str(MAX_SEQUENCE) + ')')

    return raw_data


def get_data(count_coefficient, shuffle, split_coefficient):
    raw_data = get_raw_data(count_coefficient, shuffle, True)
    voc = get_voc(raw_data)

    train, validation = split_data(raw_data, split_coefficient)
    validation_generator = DataSupplier(BATCH_SIZE, validation, voc)
    generator = DataSupplier(BATCH_SIZE, train, voc)

    print('\ndata(' + str(len(train)) + ', ' + str(len(validation)) + '),',
          'voc_size(' + str(len(voc)) + '),',
          'max_sequence(' + str(MAX_SEQUENCE) + ')\n',
          'voc(' + str(voc) + ')\n')

    return (generator, validation_generator), calculate_steps(train, validation), voc
