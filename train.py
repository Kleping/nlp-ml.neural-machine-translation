#%%

import numpy as np
import tensorflow as tf
import random as rn

#%%

batch_size = 64
epochs = 500
latent_dim = 128
# num_data = 100*batch_size
language_tag = 'en'
data_path = 'data/' + language_tag + '/encoded.txt'
model_name = 'nmt'
validation_split = .2
voc_size_source = 100
voc_size_target = 100 + 2
i_bos = voc_size_source
i_eos = voc_size_source + 1
voc_size_source += 1
voc_size_target += 1

#%%


def encode_target(text):
    return str(i_bos) + ' ' + text + ' ' + str(i_eos)


def find_max_seq_data_len(data):
    max_source_seq_len = 0
    max_target_seq_len = 0
    for sample in data:
        source, target = sample.split("\t")

        source_len = len(source.split())
        if source_len > max_source_seq_len:
            max_source_seq_len = source_len

        target_len = len(encode_target(target).split())
        if target_len > max_target_seq_len:
            max_target_seq_len = target_len

    return max_source_seq_len, max_target_seq_len


def split_data(data):
    data_validation = data[-int(validation_split * len(data)):]
    data_train = data[:int(len(data) - len(data_validation))]
    return data_train, data_validation


with open(data_path, "r", encoding="utf-8") as f:
    data = f.read().split("\n")
    # data = data[:min(len(data), num_data)]
    print(len(data))


max_source_seq_len, max_target_seq_len = find_max_seq_data_len(data)
rn.shuffle(data)
data_train, data_valid = split_data(data)

#%%


class DataSupplier(tf.keras.utils.Sequence):
    def __init__(self, batch_size, max_source_seq_len, max_target_seq_len, data, voc_size_source, voc_size_target):
        self.batch_size = batch_size
        self.data = data
        self.voc_size_source = voc_size_source
        self.voc_size_target = voc_size_target
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
            (len(source), self.max_source_seq_len, self.voc_size_source), dtype="float32"
        )
        decoder_input_data = np.zeros(
            (len(target), self.max_target_seq_len, self.voc_size_target), dtype="float32"
        )
        decoder_target_data = np.zeros(
            (len(target), self.max_target_seq_len, self.voc_size_target), dtype="float32"
        )

        for i, (source_text, target_text) in enumerate(zip(source, target)):
            for t, i_vocab in enumerate(source_text.split()):
                encoder_input_data[i, t, int(i_vocab)+1] = 1.

            # It's maybe a temporary solution
            for t in range(len(source_text.split()), self.max_source_seq_len):
                encoder_input_data[i, t, 0] = 1.

            for t, i_vocab in enumerate(target_text.split()):
                decoder_input_data[i, t, int(i_vocab)+1] = 1.
                if t > 0:
                    decoder_target_data[i, t - 1, int(i_vocab)+1] = 1.

            # It's maybe a temporary solution
            for t in range(len(target_text.split()), self.max_target_seq_len):
                decoder_input_data[i, t, 0] = 1.

            # It's maybe a temporary solution
            for t in range(len(target_text.split())-1, self.max_target_seq_len):
                decoder_target_data[i, t, 0] = 1.

        return [encoder_input_data, decoder_input_data], decoder_target_data

    @staticmethod
    def append_sample(sample, source, target):
        source_item, target_item = sample.split('\t')
        source.append(source_item)
        target.append(encode_target(target_item))
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


#%%

encoder_inputs = tf.keras.Input(shape=(None, voc_size_source))
bidirectional = tf.keras.layers.Bidirectional
encoder = bidirectional(
    tf.keras.layers.LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True
    )
)
encoder_stack_h, forward_last_h, forward_last_c, backward_last_h, backward_last_c = encoder(encoder_inputs)

encoder_last_h = tf.keras.layers.Concatenate()([forward_last_h, backward_last_h])
encoder_last_c = tf.keras.layers.Concatenate()([forward_last_c, backward_last_c])

encoder_states = [encoder_last_h, encoder_last_c]

decoder_inputs = tf.keras.Input(shape=(None, voc_size_target))

decoder = tf.keras.layers.LSTM(latent_dim*2, return_sequences=True, return_state=True)
decoder_stack_h, _, _ = decoder(decoder_inputs, initial_state=encoder_states)

context = tf.keras.layers.Attention()([decoder_stack_h, encoder_stack_h])
decoder_concat_input = tf.keras.layers.concatenate([context, decoder_stack_h])

d0 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(latent_dim, activation="relu"))(decoder_concat_input)
dense = tf.keras.layers.Dense(voc_size_target, activation='softmax')
decoder_stack_h = tf.keras.layers.TimeDistributed(dense)(d0)

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_stack_h)

#%%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.003, amsgrad=True)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

train_supplier = DataSupplier(
    batch_size,
    max_source_seq_len,
    max_target_seq_len,
    data_train,
    voc_size_source,
    voc_size_target
)

valid_supplier = DataSupplier(
    batch_size,
    max_source_seq_len,
    max_target_seq_len,
    data_valid,
    voc_size_source,
    voc_size_target
)

es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
model.fit(
    train_supplier,
    validation_data=valid_supplier,
    epochs=epochs,
    shuffle=True,
    callbacks=[es]
)

# model.save("models/" + model_name + ".h5")
