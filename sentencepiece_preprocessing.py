import sentencepiece as spm

language_tag = 'en'

# sentences = ''
# with open('data/pnc-eng.txt', "r", encoding='utf-8') as f:
#     lines = f.read().split("\n")
#     for line in lines:
#         _, target_text = line.split("\t")
#         sentences += target_text + '\n'
#
# with open('data/spm-en.txt', 'w', encoding='utf-8') as f:
#     f.write(sentences)

npt = 'data/spm-' + language_tag + '.txt'
model_prefix = 'models/spm/' + language_tag
model_type = 'bpe'
spm.SentencePieceTrainer.Train(
    '--input=' + npt + ' '
    '--model_prefix=' + model_prefix + ' '
    '--model_type=' + model_type
)

# s = spm.SentencePieceProcessor()
# s.Init(model_file='spm.model')
# for n in range(5):
#     e = s.Encode('new york', out_type=int, enable_sampling=True, alpha=.1, nbest_size=-1)
#     print(e)
#     print(s.Decode(e))
