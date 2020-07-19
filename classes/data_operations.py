import random

from classes.constant import DATA_SUFFIXES, SENTINELS, SHIFTED_SEQ_COUNT, BATCH_SIZE, MAX_SEQUENCE

from classes.DataSupplier import DataSupplier
from classes.auxiliary import get_lines, get_voc, split_data, calculate_steps, tokenize_sequence, decompose_tokens


def get_raw_data(count_coefficient, decompose, assign_max_sequence=False):
    global MAX_SEQUENCE
    raw_data = dict()
    for suffix in DATA_SUFFIXES:
        sequences = get_lines('data/normalized/eng_' + suffix + '.txt', lambda l: l[:-1])

        if assign_max_sequence:
            max_seq = max([len(tokenize_sequence(seq)) for seq in sequences])
            if max_seq > MAX_SEQUENCE:
                MAX_SEQUENCE = max_seq

        if decompose:
            sequences_count = len(sequences)
            for i in range(sequences_count):
                seq = sequences[i]
                decomposed_sequences = decompose_tokens(tokenize_sequence(seq), False)[:-1]
                [sequences.append(' '.join(tokens)) for tokens in decomposed_sequences]

        random.shuffle(sequences)
        raw_data[suffix] = sequences[:int(len(sequences) * count_coefficient)]

    if assign_max_sequence:
        MAX_SEQUENCE += len(SENTINELS) + SHIFTED_SEQ_COUNT
        print('assigned_max_sequence(' + str(MAX_SEQUENCE) + ')')

    return raw_data


def get_fit_data(count_coefficient, split_coefficient):
    raw_data = get_raw_data(count_coefficient, decompose=False)
    voc = get_voc(raw_data)

    train, validation = split_data(raw_data, split_coefficient)
    validation_generator = DataSupplier(BATCH_SIZE, validation, voc)
    generator = DataSupplier(BATCH_SIZE, train, voc)

    print('\ndata(' + str(len(train)) + ', ' + str(len(validation)) + '),',
          'voc_size(' + str(len(voc)) + '),',
          'max_sequence(' + str(MAX_SEQUENCE) + ')\n',
          'voc(' + str(voc) + ')\n')

    return (generator, validation_generator), calculate_steps(train, validation), voc
