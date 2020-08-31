import re

prohibited = ['"']
punctuation = ['!', '?', '.', ',']
unique = list()
language_tag = 'en'
input_file_name = 'fra'
output_file_name = 'paired'
pairs = list()

with open('data/' + language_tag + '/' + input_file_name + '.txt', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
    for line in lines:
        input_text, _, _ = line.split('\t')
        if input_text in unique:
            continue

        unique.append(input_text)

for target_text in unique:
    target_text = target_text.lower().translate({ord(i): None for i in ''.join(prohibited)})
    splitted_target_text = target_text.split()
    source_tokens = list()
    for token in splitted_target_text:
        if any(map(str.isdigit, token)):
            if token[-1] in punctuation:
                token = token[:-1]
        else:
            token = token.translate({ord(i): None for i in ''.join(punctuation)})
        source_tokens.append(token)
    source_text = ' '.join(source_tokens)
    pairs.append(source_text + '\t' + target_text)


def split_with_keep_delimiters(text, delimiters):
    return [
        t for t in re.split('(' + '|'.join(map(re.escape, delimiters)) + ')', text) if t is not ''
    ]


def split_source(text):
    return text.split()


def split_target(text):
    tokens = list()
    for t in text.split():
        if any(map(str.isdigit, t)):
            if t[-1] in punctuation:
                tokens.append(t[:-1])
                tokens.append(t[-1])
            else:
                tokens.append(t)
        else:
            tokens += split_with_keep_delimiters(t, punctuation)

    return tokens


def get_pairs():
    return '\n'.join(pairs)


def get_segmented_pairs():
    paired_data = list()
    unique_keys = dict()

    for line in pairs:
        target_shifted_num = 0

        source_text, target_text = line.split('\t')
        source_tokens = split_source(source_text)
        target_tokens = split_target(target_text)

        for n_word in range(len(source_tokens)):
            source_text = ' '.join(source_tokens[:(n_word + 1)])
            target_text = ''
            if target_tokens[(n_word + target_shifted_num + 1)] in punctuation:
                target_shifted_num += 1
            target_range = target_tokens[:(n_word + 1 + target_shifted_num)]
            if target_range[-1] is ',':
                target_range = target_range[:-1]
            for i, tw in enumerate(target_range):
                target_text += ('' if (i == 0) or (tw in punctuation) or (len(target_range) == i) else ' ') + tw

            pair = source_text + '\t' + target_text
            if pair not in unique_keys:
                unique_keys[pair] = ''
                paired_data.append(pair)

    return '\n'.join(paired_data)


with open('data/' + language_tag + '/' + output_file_name + '.txt', 'w', encoding='utf-8') as f:
    f.write(get_pairs())
