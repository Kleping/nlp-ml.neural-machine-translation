import numpy as np
import tensorflow as tf
import random

from classes.constant import MAX_SEQUENCE, SENTINELS, VOCABULARY_PUNCTUATION
from classes.auxiliary import tokenize_sequence, encode_seq, decompose_tokens, clothe_to, seq_to_tokens


class DataSupplier(tf.keras.utils.Sequence):
    def __init__(self, batch_size, sentences, voc):
        self.batch_size = batch_size
        self.sentences = sentences

        self.voc_size = len(voc)
        self.voc = voc
        self.d_type = 'int32'
        self.input_length = MAX_SEQUENCE

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.sentences) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sentences))
        np.random.shuffle(self.indexes)

    def get_batched_container(self):
        return np.zeros((self.batch_size, self.input_length), dtype=self.d_type)

    def __data_generation(self, indexes):
        encoder = self.get_batched_container()
        decoder = self.get_batched_container()
        output  = self.get_batched_container()

        cluster = [self.sentences[i] for i in indexes]

        for n in range(len(cluster)):
            tokens = tokenize_sequence(cluster[n])
            encoded_seq = encode_seq(tokens, self.voc)

            encoder [n] = encode_seq([i for i in tokens if i not in VOCABULARY_PUNCTUATION], self.voc)
            decoder [n] = np.insert(encoded_seq[:-1], 0, self.voc.index(SENTINELS[0]))
            output  [n] = np.insert(encoded_seq[:-1], len(tokens), self.voc.index(SENTINELS[1]))

            # these arrays are only for test a structure representation of the data
            # encoded_test = seq_to_tokens(encoder [n], self.voc)
            # decoded_test = seq_to_tokens(decoder [n], self.voc)
            # output_test  = seq_to_tokens(output  [n], self.voc)

        return [encoder, decoder], output
