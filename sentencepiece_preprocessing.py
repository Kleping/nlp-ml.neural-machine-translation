import sentencepiece as spm
from os import path

language_tag = 'en'
model_prefix = 'models/' + language_tag + '/spm'
model_file = model_prefix + '.model'
npt = 'data/' + language_tag + '/target.txt'
model_type = 'bpe'

if not path.exists(model_file):
    if not path.exists('data/' + language_tag + '/target.txt'):
        if not path.exists('data/' + language_tag + '/paired.txt'):
            print('a file by path [data/' + language_tag + '/paired.txt] does not exist')
            quit()

        target_texts = ''
        with open('data/' + language_tag + '/paired.txt', "r", encoding='utf-8') as f:
            for line in f.read().split("\n"):
                _, target_text = line.split("\t")
                target_texts += target_text + '\n'

        with open('data/' + language_tag + '/target.txt', 'w', encoding='utf-8') as f:
            f.write(target_texts)

    spm.SentencePieceTrainer.Train(
        '--input=' + npt + ' '
        '--model_prefix=' + model_prefix + ' '
        '--model_type=' + model_type
    )


def encode(raw_text, spp):
    return spp.Encode(raw_text, out_type=int, enable_sampling=True, alpha=.1, nbest_size=-1)


def decode(encoded_text, spp):
    return spp.Decode(encoded_text)


spp = spm.SentencePieceProcessor()
spp . Init(model_file=model_file)
encoded_texts = ''
with open('data/' + language_tag + '/paired.txt', "r", encoding='utf-8') as f:
    lines = f.read().split('\n')
    is_first = False
    for line in lines:
        source_text, target_text = line.split("\t")
        encoded_source = ' '.join([str(n) for n in encode(source_text, spp)])
        encoded_target = ' '.join([str(n) for n in encode(target_text, spp)])
        encoded_texts += str(('\n' if is_first else '') + encoded_source + '\t' + encoded_target)
        is_first = True

with open('data/' + language_tag + '/encoded.txt', 'w', encoding='utf-8') as f:
    f.write(encoded_texts)
