import sentencepiece as spm
from os import path, remove
import re

language_tag = 'en'
punctuation = ['!', '?', '.', ',']
model_prefix = 'models/' + language_tag + '/'
model_source_file = model_prefix + 'source.model'
model_target_file = model_prefix + 'target.model'

data_path = 'data/' + language_tag + '/'

source_path   = data_path + 'source'   + '.txt'
target_path   = data_path + 'target'   + '.txt'
original_path = data_path + 'original' + '.txt'
paired_path   = data_path + 'paired'   + '.txt'
encoded_path  = data_path + 'encoded'  + '.txt'

npt = 'data/' + language_tag + '/'
model_type = 'bpe'
batch_size = 64
num_data = 100*batch_size
voc_size = 100

if not path.exists(original_path):
    print('a file by path [' + original_path + '] does not exist')
    quit()

if not path.exists(paired_path):
    paired_data = list()
    unique_keys = dict()

    def split_source(source_text):
        return source_text.split()

    def split_target(target_text):
        target_tokens = list()
        [
            [
                target_tokens.append(w) for w in i.split() if w is not ''
            ]
            for i in re.split('(' + '|'.join(map(re.escape, punctuation)) + ')', target_text)
        ]
        return target_tokens

    with open(original_path, "r", encoding='utf-8') as f:
        lines = f.read().split('\n')
        lines_counter = 0
        last_percent = 0
        ln = len(lines)
        for line in lines:
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
                for i, tw in enumerate(target_range):
                    target_text += ('' if (i == 0) or (tw in punctuation) or (len(target_range) == i) else ' ') + tw

                pair = source_text + '\t' + target_text
                if pair not in unique_keys:
                    unique_keys[pair] = ''
                    paired_data.append(pair)

            lines_counter += 1
            percent = int(lines_counter / ln * 100)
            if percent != last_percent:
                print(percent, '/', '100')
                last_percent = percent

    with open(paired_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(paired_data))

if (not path.exists(source_path)) or (not path.exists(target_path)):
    source_texts = list()
    target_texts = list()
    with open(paired_path, "r", encoding='utf-8') as f:
        for line in f.read().split("\n"):
            source_text, target_text = line.split('\t')
            source_texts.append(source_text)
            target_texts.append(target_text)

    with open(source_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(source_texts))

    with open(target_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(target_texts))

spm.SentencePieceTrainer.Train(
    '--input=' + source_path + ' '
    '--model_prefix=' + model_prefix + 'source '
    '--model_type=' + model_type + ' '
    '--vocab_size=' + str(voc_size)
)

spm.SentencePieceTrainer.Train(
    '--input=' + target_path + ' '
    '--model_prefix=' + model_prefix + 'target '
    '--model_type=' + model_type + ' '
    '--vocab_size=' + str(voc_size)
)


def encode(raw_text, spp):
    return spp.Encode(raw_text, out_type=int, enable_sampling=False, alpha=.1, nbest_size=-1)


def decode(encoded_text, spp):
    return spp.Decode(encoded_text)


spp_source = spm.SentencePieceProcessor()
spp_target = spm.SentencePieceProcessor()
spp_source . Init(model_file=model_source_file)
spp_target . Init(model_file=model_target_file)
encoded_texts = ''

with open(paired_path, "r", encoding='utf-8') as f:
    lines = f.read().split('\n')
    is_first = False
    for line in lines:
        source_text, target_text = line.split("\t")
        encoded_source = ' '.join([str(n) for n in encode(source_text, spp_source)])
        encoded_target = ' '.join([str(n) for n in encode(target_text, spp_target)])
        encoded_texts += str(('\n' if is_first else '') + encoded_source + '\t' + encoded_target)
        is_first = True

with open(encoded_path, 'w', encoding='utf-8') as f:
    f.write(encoded_texts)
