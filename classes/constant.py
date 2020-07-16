
DISTRIBUTION_DATA_COUNT = [20 * 1000, 40 * 1000, 60 * 1000]
VOCABULARY_PUNCTUATION = ['!', '?', '.', ',']
DATA_SUFFIXES = ['one', 'two', 'three']
SENTINELS = ['^', '~']

MODEL_NAME = 'encoder_decoder_model'
MODEL_PATH = 'models/' + MODEL_NAME + '.h5'
DATA_NAME = 'data/original.txt'

SHIFTED_SEQ_COUNT = 1
LATENT_DIMENSIONS = 128
BATCH_SIZE = 64
EPOCHS = 1
MAX_SEQUENCE = 300
ACCEPTED_DIFF = .01
