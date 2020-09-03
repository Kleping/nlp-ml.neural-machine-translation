import numpy as np
import tensorflow as tf
import random as rn
import re
import json

validation_split = .3

epochs = 100
batch_size = 128
coefficient = 500
latent_dim = 32

model_name = 'nmt_{}_{}_{}_{}'.format(epochs, batch_size, coefficient, latent_dim)
num_data = coefficient*batch_size

language_tag = 'en'
data_path = 'data/{}/paired.txt'.format(language_tag)
model_path = 'models/{}/{}.h5'.format(language_tag, model_name)

punctuation = ['!', '?', '.', ',']
sentinels = ['<BOS>', '<EOS>']
OOV_TOKEN = '<OOV>'
PAD_TOKEN = '<PAD>'


def split_with_keep_delimiters(text, delimiters):
    return [
        token for token in re.split('(' + '|'.join(map(re.escape, delimiters)) + ')', text) if token is not ''
    ]


def tokenize_text(text):
    tokens = list()
    for token in text.split():
        if any(map(str.isdigit, token)):
            if token[-1] in punctuation:
                tokens.append(token[:-1])
                tokens.append(token[-1])
            else:
                tokens.append(token)
        else:
            [tokens.append(splitted_token) for splitted_token in split_with_keep_delimiters(token, punctuation)]

    return tokens


def encode_target(text):
    return '{} {} {}'.format(sentinels[0], text, sentinels[1])


def get_max_sample_length(data):
    max_source = 0
    max_target = 0
    for sample in data:
        source, target = sample.split('\t')

        source_len = len(tokenize_text(source))
        if source_len > max_source:
            max_source = source_len

        target_len = len(tokenize_text(encode_target(target)))
        if target_len > max_target:
            max_target = target_len

    return max_source, max_target


def split_data(data):
    data_validation = data[-int(validation_split * len(data)):]
    data_train = data[:int(len(data) - len(data_validation))]
    return data_train, data_validation[:len(data_validation)//2]


def get_voc_from_data(data):
    bag_of_words = list()
    for sample in data:
        source, _ = sample.split('\t')
        tokenized_source = tokenize_text(source)
        [bag_of_words.append(token) for token in tokenized_source if token not in bag_of_words]
    source_v = (bag_of_words + [OOV_TOKEN])
    target_v = (bag_of_words + sentinels + punctuation + [OOV_TOKEN])
    rn.shuffle(source_v)
    rn.shuffle(target_v)
    return ([PAD_TOKEN] + source_v), ([PAD_TOKEN] + target_v)


def serialize_and_write_config(source, target, max_source_len, max_target_len, history):
    config = {
        'source': source,
        'target': target,
        'max_source_len': max_source_len,
        'max_target_len': max_target_len,
        'history': history
    }
    with open('models/{}/{}.config'.format(language_tag, model_name), 'w') as config_file:
        json.dump(config, config_file, indent=4)


def compile_model(m):
    optimizer = tf.keras.optimizers.Adam()

    m.compile(
        optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']
    )
    return m


with open(data_path, 'r', encoding='utf-8') as f:
    data = f.read().split('\n')[:num_data]

rn.shuffle(data)
data_train, data_valid = split_data(data)
max_source_len, max_target_len = get_max_sample_length(data_train + data_valid)
source_voc, target_voc = get_voc_from_data(data_train)

print()
print('len(data):', len(data))
print('max_source_len:', max_source_len)
print('max_target_len:', max_target_len)
# print('source_voc:', source_voc)
# print('target_voc:', target_voc)
print('len(source_voc):', len(source_voc))
print('len(target_voc):', len(target_voc))
print()


class DataSupplier(tf.keras.utils.Sequence):
    def __init__(self, batch_size, max_source_seq_len, max_target_seq_len, data, source_voc, target_voc):
        self.batch_size = batch_size
        self.data = data
        self.source_voc = source_voc
        self.target_voc = target_voc
        self.max_source_seq_len = max_source_seq_len
        self.max_target_seq_len = max_target_seq_len
        if self.data is not None:
            rn.shuffle(self.data)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, ndx):
        source, target = self.extract_batch(ndx, self.batch_size, self.data)
        return self.encode_data(source, target)

    def on_epoch_end(self):
        rn.shuffle(self.data)

    # secondary auxiliary methods
    @staticmethod
    def get_token_index(voc, token):
        if token in voc:
            voc_i = voc.index(token)
        else:
            voc_i = voc.index(OOV_TOKEN)
        return voc_i

    def encode_data(self, source, target):
        encoder_input_data = np.zeros(
            (len(source), self.max_source_seq_len), dtype='int32'
        )
        decoder_input_data = np.zeros(
            (len(target), self.max_target_seq_len), dtype='int32'
        )
        decoder_target_data = np.zeros(
            (len(target), self.max_target_seq_len), dtype='int32'
        )

        for i, (source_text, target_text) in enumerate(zip(source, target)):
            for t, token in enumerate(tokenize_text(source_text)):
                encoder_input_data[i, t] = self.get_token_index(self.source_voc, token)

            for t, token in enumerate(tokenize_text(target_text)):
                token_ndx = self.get_token_index(self.target_voc, token)
                decoder_input_data[i, t] = token_ndx
                if t > 0:
                    decoder_target_data[i, t - 1] = token_ndx

            # It's only a debug representation
            # print('\n{}'.format(' '.join([source_voc[token] for token in encoder_input_data[i]])))
            # print('{}'.format(' '.join([target_voc[token] for token in decoder_target_data[i]])))

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


encoder_inputs = tf.keras.Input(shape=(None,))
encoder_emb = tf.keras.layers.Embedding(len(source_voc), latent_dim)(encoder_inputs)

encoder = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(
        latent_dim,
        return_sequences=True,
        recurrent_dropout=.1
    )
)

encoder_secondary = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(
        latent_dim*2,
        return_sequences=True,
        return_state=True,
        recurrent_dropout=.1
    )
)

encoder_stack_h = encoder(encoder_emb)
encoder_secondary_stack_h, forward_last_h, forward_last_c, backward_last_h, backward_last_c \
    = encoder_secondary(encoder_stack_h)

encoder_last_h = tf.keras.layers.Concatenate()([forward_last_h, backward_last_h])
encoder_last_c = tf.keras.layers.Concatenate()([forward_last_c, backward_last_c])

encoder_states = [encoder_last_h, encoder_last_c]

decoder_inputs = tf.keras.Input(shape=(None,))
decoder_emb = tf.keras.layers.Embedding(len(target_voc), latent_dim)(decoder_inputs)
decoder = tf.keras.layers.LSTM(latent_dim*4, return_sequences=True, return_state=True)
decoder_stack_h, _, _ = decoder(decoder_emb, initial_state=encoder_states)

context = tf.keras.layers.Attention()([decoder_stack_h, encoder_secondary_stack_h])
decoder_concat_input = tf.keras.layers.concatenate([context, decoder_stack_h])

dense = tf.keras.layers.Dense(len(target_voc), activation='softmax')
decoder_stack_h = tf.keras.layers.TimeDistributed(dense)(decoder_concat_input)

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_stack_h)
model = compile_model(model)

train_supplier = DataSupplier(
    batch_size,
    max_source_len,
    max_target_len,
    data_train,
    source_voc,
    target_voc
)

valid_supplier = DataSupplier(
    batch_size,
    max_source_len,
    max_target_len,
    data_valid,
    source_voc,
    target_voc
)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

history = model.fit(
    train_supplier,
    validation_data=valid_supplier,
    epochs=epochs,
    shuffle=True,
    callbacks=[es]
).history

model.save(model_path)
serialize_and_write_config(source_voc, target_voc, max_source_len, max_target_len, history)
