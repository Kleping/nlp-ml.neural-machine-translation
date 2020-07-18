import numpy as np
import tensorflow as tf

from classes.constant import MAX_SEQUENCE, SENTINELS, VOCABULARY_PUNCTUATION
from classes.auxiliary import tokenize_sequence, encode_seq, seq_to_tokens


class DataSupplier(tf.keras.utils.Sequence):
    def __init__(self, batch_size, sentences, voc):
        self.batch_size = batch_size
        self.sentences = sentences

        self.voc_size = len(voc)
        self.voc = voc

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.sentences) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sentences))
        np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        encoded_input = np.zeros((self.batch_size, MAX_SEQUENCE, self.voc_size), dtype='float32')
        decoded_input = np.zeros((self.batch_size, MAX_SEQUENCE, self.voc_size), dtype='float32')
        decoded_output = np.zeros((self.batch_size, MAX_SEQUENCE, self.voc_size), dtype='float32')

        cluster = [self.sentences[i] for i in indexes]

        for n in range(len(cluster)):
            tokens = tokenize_sequence(cluster[n])
            tokens.insert(0, SENTINELS[0])
            tokens.append(SENTINELS[1])

            for i in range(len(tokens)):
                c = self.voc.index(tokens[i])
                # a number of sample, an index of position in the current sentence,
                # an index of character in the vocabulary
                decoded_output[n, i, c] = 1.

                # a number of sample, an index of shifted position in the current sentence,
                # an index of character in the vocabulary
                decoded_input[n, i + 1, c] = 1.

            seq_without_punctuation = [i for i in tokens if i not in VOCABULARY_PUNCTUATION]
            encoded_input[n] = encode_seq(seq_without_punctuation, self.voc)

        return [encoded_input, decoded_input], decoded_output
