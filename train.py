#%%

import numpy as np
import tensorflow as tf
import random as rn

#%%

batch_size = 64  # Batch size for training.
validation_batch_size = 40
epochs = 1  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_single_cluster_pairs = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'data/pnc-eng.txt'
model_name = 'seq2seq_with_attention'
validation_split = .2
num_limit_clusters = 2
num_train_cluster = int(num_single_cluster_pairs * (1. - validation_split))
num_valid_cluster = num_single_cluster_pairs - num_train_cluster

#%%

# Vectorization of the data.
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
    pairs = f.read().split("\n")
    pairs = pairs[:min(len(pairs), num_limit_clusters * num_single_cluster_pairs)]
num_total_pairs = len(pairs)
num_residue_pairs = num_total_pairs % num_single_cluster_pairs
num_clusters = num_total_pairs // num_single_cluster_pairs + (1 if num_residue_pairs != 0 else 0)
max_source_seq_len = 0
max_target_seq_len = 0
for pair in pairs:
    source_text, target_text = pair.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"

    for char in source_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

    if len(source_text) > max_source_seq_len:
        max_source_seq_len = len(source_text)
    if len(target_text) > max_target_seq_len:
        max_target_seq_len = len(target_text)

rn.shuffle(pairs)
pairs_validation = pairs[-int(validation_split * num_total_pairs):]
pairs_train = pairs[:int(num_total_pairs-len(pairs_validation))]

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

num_source_tokens = len(input_characters)
num_target_tokens = len(target_characters)

print("Number of residue pairs:", num_residue_pairs)
print("Number of clusters:", num_clusters)
print("Number of samples:", num_total_pairs)
print("Number of unique input tokens:", num_source_tokens)
print("Number of unique output tokens:", num_target_tokens)
print("Max sequence length:", max_source_seq_len)


def get_data_cluster(i_cluster, n_cluster, pairs):
    source_texts = []
    target_texts = []
    cluster_from = i_cluster * n_cluster
    cluster_to = min(cluster_from + n_cluster, num_total_pairs)

    if i_cluster == 0:
        rn.shuffle(pairs)

    for pair in pairs[cluster_from: cluster_to]:
        source_text, target_text = pair.split("\t")
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = "\t" + target_text + "\n"
        source_texts.append(source_text)
        target_texts.append(target_text)

    if cluster_to % n_cluster != 0:
        pairs_residue = rn.sample(pairs[:cluster_from], n_cluster - num_residue_pairs)
        for pair in pairs_residue:
            source_text, target_text = pair.split("\t")
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = "\t" + target_text + "\n"
            source_texts.append(source_text)
            target_texts.append(target_text)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(source_texts), max_source_seq_len, num_source_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(target_texts), max_target_seq_len, num_target_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(target_texts), max_target_seq_len, num_target_tokens), dtype="float32"
    )

    for i, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
        for t, char in enumerate(source_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0

    return [encoder_input_data, decoder_input_data], decoder_target_data

#%%

# Define an input sequence and process it.
encoder_inputs = tf.keras.Input(shape=(None, num_source_tokens))
encoder = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_stack_h, encoder_last_h, encoder_last_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [encoder_last_h, encoder_last_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = tf.keras.Input(shape=(None, num_target_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_stack_h, _, _ = decoder(decoder_inputs, initial_state=encoder_states)

context = tf.keras.layers.Attention()([decoder_stack_h, encoder_stack_h])
decoder_concat_input = tf.keras.layers.concatenate([context, decoder_stack_h])

dense = tf.keras.layers.Dense(num_target_tokens, activation='softmax')
decoder_stack_h = tf.keras.layers.TimeDistributed(dense)(decoder_concat_input)

# decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation="softmax")
# decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_stack_h)

#%%

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
# model.fit(
#     [encoder_input_data, decoder_input_data],
#     decoder_target_data,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_split=validation_split,
# )

for epoch in range(epochs):
    for i_cluster in range(num_clusters):
        if i_cluster == 0:
            print(str(epoch+1) + '/' + str(epochs), 'epochs')
        x, Y = get_data_cluster(i_cluster, num_train_cluster, pairs_train)
        x_validation, Y_validation = get_data_cluster(i_cluster, num_valid_cluster, pairs_validation)
        model.fit(
            x, Y,
            validation_data=(x_validation, Y_validation),
            batch_size=batch_size,
            epochs=1,
            shuffle=True,
        )

# Save model
model.save("models/" + model_name + ".h5")
