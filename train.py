#%%

import numpy as np
import tensorflow as tf
import random as rn

#%%

batch_size = 64  # Batch size for training.
validation_batch_size = 40
epochs = 1  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = "data/pnc-eng.txt"
model_name = 'seq2seq_with_attention'
validation_split = .2
num_limit_clusters = 2

#%%

# Vectorize the data.
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
    lines = lines[:min(len(lines), num_limit_clusters * num_samples)]
num_total_samples = len(lines)
num_residue_samples = num_total_samples % num_samples
num_clusters = num_total_samples // num_samples + (1 if num_residue_samples != 0 else 0)
max_seq_length = 0
for line in lines:
    input_text, target_text = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
    if max(len(input_text), len(target_text)) > max_seq_length:
        max_seq_length = max(len(input_text), len(target_text))

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

print("Number of residue samples:", num_residue_samples)
print("Number of clusters:", num_clusters)
print("Number of samples:", num_total_samples)
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length:", max_seq_length)


def get_data_cluster(n_cluster):
    input_texts = []
    target_texts = []
    cluster_from = n_cluster*num_samples
    cluster_to = min(cluster_from+num_samples, num_total_samples)

    if n_cluster == 0:
        rn.shuffle(lines)

    for line in lines[cluster_from: cluster_to]:
        input_text, target_text = line.split("\t")
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = "\t" + target_text + "\n"
        input_texts.append(input_text)
        target_texts.append(target_text)

    if cluster_to % num_samples != 0:
        samples = rn.sample(lines[:cluster_from], num_samples - num_residue_samples)
        for line in samples:
            input_text, target_text = line.split("\t")
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = "\t" + target_text + "\n"
            input_texts.append(input_text)
            target_texts.append(target_text)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_seq_length, num_encoder_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_seq_length, num_decoder_tokens), dtype="float32"
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
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

    return encoder_input_data, decoder_input_data, decoder_target_data

#%%

# Define an input sequence and process it.
encoder_inputs = tf.keras.Input(shape=(None, num_encoder_tokens))
encoder = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_stack_h, encoder_last_h, encoder_last_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [encoder_last_h, encoder_last_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = tf.keras.Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_stack_h, _, _ = decoder(decoder_inputs, initial_state=encoder_states)

context = tf.keras.layers.Attention()([decoder_stack_h, encoder_stack_h])
decoder_concat_input = tf.keras.layers.concatenate([context, decoder_stack_h])

dense = tf.keras.layers.Dense(num_decoder_tokens, activation='softmax')
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
    for i_cl in range(num_clusters):
        if i_cl == 0:
            print(str(epoch+1) + '/' + str(epochs))
        encoder_input_data, decoder_input_data, decoder_target_data = get_data_cluster(i_cl)
        model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=batch_size,
            epochs=1,
            validation_split=validation_split,
            shuffle=True,
        )
# Save model
model.save("models/" + model_name + ".h5")
