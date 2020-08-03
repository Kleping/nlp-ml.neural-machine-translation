import numpy as np
import tensorflow as tf
import random as rn


class DataSupplier(tf.keras.utils.Sequence):
    def __init__(self, batch_size, max_source_seq_len, max_target_seq_len, data, voc_size):
        self.batch_size = batch_size
        self.data = data
        self.voc_size = voc_size
        self.max_source_seq_len = max_source_seq_len
        self.max_target_seq_len = max_target_seq_len
        rn.shuffle(self.data)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, ndx):
        source, target = self.extract_batch(ndx, self.batch_size, self.data)
        return self.encode_data(source, target)

    def on_epoch_end(self):
        rn.shuffle(self.data)

    # secondary auxiliary methods
    def encode_data(self, source, target):
        encoder_input_data = np.zeros(
            (len(source), self.max_source_seq_len, self.vocab_size), dtype="float32"
        )
        decoder_input_data = np.zeros(
            (len(target), self.max_target_seq_len, self.vocab_size), dtype="float32"
        )
        decoder_target_data = np.zeros(
            (len(target), self.max_target_seq_len, self.vocab_size), dtype="float32"
        )

        for i, (source_text, target_text) in enumerate(zip(source, target)):
            for t, i_vocab in enumerate(source_text.split()):
                encoder_input_data[i, t, int(i_vocab)] = 1.

            for t, i_vocab in enumerate(target_text.split()):
                decoder_input_data[i, t, int(i_vocab)] = 1.
                if t > 0:
                    decoder_target_data[i, t - 1, int(i_vocab)] = 1.

        return [encoder_input_data, decoder_input_data], decoder_target_data

    def append_sample(self, sample, source, target):
        source_item, target_item = sample.split('\t')
        source.append(source_item)
        target.append(self.encode_target(target_item))
        return source, target

    def extract_batch(self, ndx, batch_size, data):
        source = []
        target = []
        ndx_from = ndx * batch_size
        ndx_to = min(ndx * batch_size + batch_size, len(data))

        for sample in data[ndx_from: ndx_to]:
            source, target = self.append_sample(sample, source, target)

        if ndx_to % batch_size != 0:
            for sample in rn.sample(data[:ndx_from], batch_size - len(data) % batch_size):
                source, target = self.append_sample(sample, source, target)

        return source, target
