import numpy as np
import keras as k
from classes import constant

from classes.auxiliary import split_with_keep_delimiters


class DataSupplier(k.utils.Sequence):
    def __init__(self, batch_size, sentences, voc, voc_size):
        self.batch_size = batch_size
        self.sentences = sentences

        self.voc_size = voc_size
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
        encoded_input = np.zeros((self.batch_size, constant.MAX_SEQUENCE, self.voc_size), dtype='float32')
        decoded_input = np.zeros((self.batch_size, constant.MAX_SEQUENCE, self.voc_size), dtype='float32')
        decoded_output = np.zeros((self.batch_size, constant.MAX_SEQUENCE, self.voc_size), dtype='float32')

        cluster = [self.sentences[i] for i in indexes]
        delimiters = [' ']

        for n in range(len(cluster)):
            string = cluster[n]
            words = split_with_keep_delimiters(string, delimiters)
            words.insert(0, constant.SENTINELS[0])
            words.append(constant.SENTINELS[1])

            for i in range(len(words)):
                c = self.voc.index(words[i])
                # a number of sample, an index of position in the current sentence,
                # an index of character in the vocabulary
                decoded_output[n, i, c] = 1.

                # a number of sample, an index of shifted position in the current sentence,
                # an index of character in the vocabulary
                decoded_input[n, i + 1, c] = 1.

            sentence_without_punctuation = [i for i in cluster[n].split() if i not in constant.VOCABULARY_PUNCTUATION]
            [sentence_without_punctuation.insert(i, ' ') for i in range(1, len(sentence_without_punctuation)*2 - 1, 2)]
            for i in range(len(sentence_without_punctuation)):
                c = self.voc.index(sentence_without_punctuation[i])
                # a number of sample, an index of position in the current sentence,
                # an index of character in the vocabulary
                encoded_input[n, i, c] = 1.

        return [encoded_input, decoded_input], decoded_output
