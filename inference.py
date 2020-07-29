#%%

import numpy as np
import tensorflow as tf
import random as rn

#%%

latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 40000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = "data/pnc-eng.txt"
model_name = 'bidirectional_seq2seq_with_attention'

#%%

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
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

#%%

# Define sampling models
# Restore the model and construct the encoder and decoder.
model = tf.keras.models.load_model("models/" + model_name + ".h5")
encoder_input = model.inputs[0]
encoder_output, forward_last_h, forward_last_c, backward_last_h, backward_last_c = model.layers[1].output

encoder_last_h = tf.keras.layers.Concatenate()([forward_last_h, backward_last_h])
encoder_last_c = tf.keras.layers.Concatenate()([forward_last_c, backward_last_c])

encoder_model = tf.keras.Model(encoder_input, [encoder_output] + [encoder_last_h, encoder_last_c])

decoder_input = model.inputs[1]
decoder_state_input_h = tf.keras.Input(shape=(latent_dim*2,), name="input_3")
decoder_state_input_c = tf.keras.Input(shape=(latent_dim*2,), name="input_4")
encoder_stack_h = tf.keras.Input(shape=(None, latent_dim*2,), name="input_5")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder = model.layers[5]
decoder_stack_h, decoder_last_h, decoder_last_c = decoder(
    decoder_input, initial_state=decoder_states_inputs
)
decoder_attention = model.layers[6]
decoder_concatenate = model.layers[7]
decoder_dense = model.layers[8]

decoder_last = [decoder_last_h, decoder_last_c]
context = decoder_attention([decoder_stack_h, encoder_stack_h])

decoder_stack_h = decoder_dense(tf.keras.layers.concatenate([context, decoder_stack_h]))
decoder_model = tf.keras.Model(
    [decoder_input, encoder_stack_h, decoder_states_inputs],
    [decoder_stack_h] + decoder_last
)
print(decoder_model.summary())

#%%

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    encoder_outputs, h, c = encoder_model.predict(input_seq)
    states_value = [h, c]
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        # scores = tf.keras.layers.Attention()._calculate_scores(states_value, encoder_outputs)
        output_tokens, h, c = decoder_model.predict([target_seq, encoder_outputs, states_value])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence


for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    i = seq_index * 100
    input_seq = encoder_input_data[i : i + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[i])
    print("Decoded sentence:", decoded_sentence)
