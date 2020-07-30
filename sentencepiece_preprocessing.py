import sentencepiece as spm
from os import path, remove

language_tag = 'en'
model_prefix = 'models/' + language_tag + '/'
model_source_file = model_prefix + 'source.model'
model_target_file = model_prefix + 'target.model'
npt = 'data/' + language_tag + '/'
model_type = 'bpe'


if not path.exists('data/' + language_tag + '/paired.txt'):
    print('a file by path [data/' + language_tag + '/paired.txt] does not exist')
    quit()

target_texts = ''
source_texts = ''
with open('data/' + language_tag + '/paired.txt', "r", encoding='utf-8') as f:
    for line in f.read().split("\n"):
        source_text, target_text = line.split("\t")
        source_texts += source_text + '\n'
        target_texts += target_text + '\n'

with open('data/' + language_tag + '/source.txt', 'w', encoding='utf-8') as f:
    f.write(source_texts)

with open('data/' + language_tag + '/target.txt', 'w', encoding='utf-8') as f:
    f.write(target_texts)

spm.SentencePieceTrainer.Train(
    '--input=' + npt + 'source.txt '
    '--model_prefix=' + model_prefix + 'source '
    '--model_type=' + model_type
)

spm.SentencePieceTrainer.Train(
    '--input=' + npt + 'target.txt '
    '--model_prefix=' + model_prefix + 'target '
    '--model_type=' + model_type
)

remove('data/' + language_tag + '/source.txt')
remove('data/' + language_tag + '/target.txt')


def encode(raw_text, spp):
    return spp.Encode(raw_text, out_type=int, enable_sampling=True, alpha=.1, nbest_size=-1)


def decode(encoded_text, spp):
    return spp.Decode(encoded_text)


spp_source = spm.SentencePieceProcessor()
spp_target = spm.SentencePieceProcessor()
spp_source . Init(model_file=model_source_file)
spp_target . Init(model_file=model_target_file)
encoded_texts = ''

with open('data/' + language_tag + '/paired.txt', "r", encoding='utf-8') as f:
    lines = f.read().split('\n')
    is_first = False
    for line in lines:
        source_text, target_text = line.split("\t")
        encoded_source = ' '.join([str(n) for n in encode(source_text, spp_source)])
        encoded_target = ' '.join([str(n) for n in encode(target_text, spp_target)])
        encoded_texts += str(('\n' if is_first else '') + encoded_source + '\t' + encoded_target)
        is_first = True

with open('data/' + language_tag + '/encoded.txt', 'w', encoding='utf-8') as f:
    f.write(encoded_texts)
