import random

from classes.constant import DISTRIBUTION_DATA_COUNT, DATA_SUFFIXES, VOCABULARY_PUNCTUATION
from classes.auxiliary import get_lines, split_with_keep_delimiters


def normalize_data(input_path, output_dir):
    data = list()

    loaded_data = get_lines(input_path,
                            lambda l: split_with_keep_delimiters(l, VOCABULARY_PUNCTUATION[:3])[:-1])

    for i, el in enumerate(loaded_data):
        if not isinstance(el, list):
            continue

        for j in range(0, len(el), 2):
            if j == 0:
                loaded_data[i] = el[j] + el[j + 1]
            else:
                loaded_data.append(el[j][1:] + el[j + 1])

    rest_count = len(loaded_data) - sum(DISTRIBUTION_DATA_COUNT)
    indexes = list(range(len(loaded_data) - 1))
    random.shuffle(indexes)
    n_one = indexes[:DISTRIBUTION_DATA_COUNT[0]]
    n_two = indexes[DISTRIBUTION_DATA_COUNT[0]: sum(DISTRIBUTION_DATA_COUNT[:2])]
    n_three = indexes[sum(DISTRIBUTION_DATA_COUNT[:2]):sum(DISTRIBUTION_DATA_COUNT[:3])]

    while len(n_one) != 0:
        data.append(loaded_data[n_one.pop(0)])
    with open(output_dir + DATA_SUFFIXES[0] + '.txt', "w", encoding='utf-8') as f:
        f.write('\n'.join(data))
    data.clear()

    while len(n_two) != 0:
        a = str(loaded_data[n_two.pop(0)])
        b = str(loaded_data[n_two.pop(0)])
        data.append(a + ' ' + b)
    with open(output_dir + DATA_SUFFIXES[1] + '.txt', "w", encoding='utf-8') as f:
        f.write('\n'.join(data))
    data.clear()

    while len(n_three) != 0:
        a = str(loaded_data[n_three.pop(0)])
        b = str(loaded_data[n_three.pop(0)])
        c = str(loaded_data[n_three.pop(0)])
        data.append(a + ' ' + b + ' ' + c)
    with open(output_dir + DATA_SUFFIXES[2] + '.txt', "w", encoding='utf-8') as f:
        f.write('\n'.join(data))
    data.clear()


def weak_data(path, delimiters, delta):
    data_incomplete = list()
    data_loaded = get_lines(path, lambda l: split_with_keep_delimiters(l, delimiters)[:-1])

    for i, el in enumerate(data_loaded):
        concatenated_item = list()
        for j in range(0, len(el), 2):
            concatenated_item.append(el[j] + el[j + 1])
        data_loaded[i] = concatenated_item

    len_loaded_data = len(data_loaded)
    count = int(len_loaded_data * delta)
    while len(data_incomplete) != count:
        sentences = data_loaded[random.randint(0, len_loaded_data - 1)]
        sentence = sentences[-1].split()
        n = random.randint(1, len(sentence) - 1)
        weak_sentence = ''
        if len(sentences) == 1:
            weak_sentence = ' '.join(sentence[:n])
        else:
            weak_sentence = ''.join(sentences[:-1]) + ' ' + ' '.join(sentence[:n])

        if weak_sentence[-1] in VOCABULARY_PUNCTUATION:
            weak_sentence = weak_sentence[:-2]

        if weak_sentence not in data_incomplete:
            data_incomplete.append(weak_sentence)

    return data_incomplete


def weaken_data(label_name, input_dir, output_dir):
    delimiters = VOCABULARY_PUNCTUATION[:3]
    for suffix in DATA_SUFFIXES:
        incomplete_data = weak_data(input_dir + label_name + suffix + '.txt', delimiters, .25)
        with open(output_dir + label_name + suffix + '.txt', "w", encoding='utf-8') as f:
            f.write('\n'.join(incomplete_data))


def reduction_noise_from_data(original_path, label_name):
    data = list()
    with open(original_path + label_name + '_original.txt', "r", encoding='utf-8') as rf:
        for line in rf.readlines():
            sentence = line.split("\t")[0]
            sentence = sentence.replace('"', '').lower()
            data.append(sentence)

    data = list(set(data))

    with open(original_path + label_name + '_de_noised.txt', 'w', encoding='utf-8') as wf:
        wf.write('\n'.join(data))


# Before of an identify parts speech
# reduction_noise_from_data('data/', 'eng')

# After a successful identified parts speech
# normalize_data('data/eng_analyzed.txt', 'data/normalized/eng_')
weaken_data('eng_', 'data/normalized/', 'data/incomplete/')
