'''Word-based sequence to sequence RNN translator that trains on a parallel corpus of 
ten Hebrew Torahs on one side and ten different English translations on the 
other. Built primarily by Justin Barber as an extensive revision of Keras author
Francois Chollet's character-based model trained on a simple French-English corpus.
'''
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import re

BATCH_SIZE = 8  # Batch size for training.
EPOCHS = 100  # Number of EPOCHS to train for.
LATENT_DIM = 256  # Latent dimensionality of the encoding space.
NUM_SAMPLES = 30000  # Number of samples to train on.
# Path to the data txt file on disk.
DATA_PATH = 'tabbed_heb_eng_corpus2.txt'

# Vectorize the data.
input_texts = []
target_texts = []
input_words = set()
target_words = set()
lines = open(DATA_PATH, encoding='utf-8').read().lower().split('\n')
for line in lines[: min(NUM_SAMPLES, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    input_text = re.findall(r'[\w\-]+', input_text)
    target_text = re.findall(r'[\w\-]+', target_text)
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = ['\t'] + target_text + ['\n']
    input_texts.append(input_text)
    target_texts.append(target_text)
    for token in input_text:
        if token not in input_words:
            input_words.add(token)
    for token in target_text:
        if token not in target_words:
            target_words.add(token)

input_words = sorted(list(input_words))
target_words = sorted(list(target_words))
num_hebrew_tokens = len(input_words)
num_english_tokens = len(target_words)
max_hebrew_seq_length = max([len(txt) for txt in input_texts])
max_english_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_hebrew_tokens)
print('Number of unique output tokens:', num_english_tokens)
print('Max sequence length for inputs:', max_hebrew_seq_length)
print('Max sequence length for outputs:', max_english_seq_length)

input_token_index = dict(
    [(token, i) for i, token in enumerate(input_words)])
target_token_index = dict(
    [(token, i) for i, token in enumerate(target_words)])

### How do the two (input / hebrewand target / english) become three here?
### Does the target / english become english_input and english_target?
### Answer: hebrew_input is hebrew and english_input is english in data,
### and the english_target is all we care about it producing as translatio
### output.
hebrew_input_data = np.zeros(
    (len(input_texts), max_hebrew_seq_length, num_hebrew_tokens),
    dtype='float32')
english_input_data = np.zeros(
    (len(input_texts), max_english_seq_length, num_english_tokens),
    dtype='float32')
english_target_data = np.zeros(
    (len(input_texts), max_english_seq_length, num_english_tokens),
    dtype='float32')

# one-hot encode the data, putting each word into a 1 in the above matrices
for vs_index, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for heb_word_index, token in enumerate(input_text):
        hebrew_input_data[vs_index, heb_word_index, input_token_index[token]] = 1.
    for eng_word_index, token in enumerate(target_text):
        # english_target_data is ahead of english_input_data by one timestep
        english_input_data[vs_index, eng_word_index, target_token_index[token]] = 1.
        if eng_word_index > 0:
            # english_target_data will be ahead by one timestep
            # and will not include the start character.
            english_target_data[vs_index, eng_word_index - 1, target_token_index[token]] = 1.

# Define an input sequence and process it.
hebrew_inputs = Input(shape=(None, num_hebrew_tokens))
hebrew_lstm = LSTM(LATENT_DIM, return_state=True)
# When called, it will give three numbers:
hebrew_outputs, hidden_state, cell_state = hebrew_lstm(hebrew_inputs)
# We discard `hebrew_outputs` and only keep the states.
hebrew_states = [hidden_state, cell_state]

# Set up the english, using `hebrew_states` as initial state.
english_inputs = Input(shape=(None, num_english_tokens))
# We set up our english to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
english_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
english_outputs, _, _ = english_lstm(english_inputs,
                                     initial_state=hebrew_states)
english_dense = Dense(num_english_tokens, activation='softmax')
english_outputs = english_dense(english_outputs)

# Define the model that will turn
# `hebrew_input_data` & `english_input_data` into `english_target_data`
model = Model([hebrew_inputs, english_inputs], english_outputs)

print(model.summary())

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([hebrew_input_data, english_input_data], english_target_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2)

# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial english state
# 2) run one step of english with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models - hebrew_model (intakes the hebrew) and 
# english_model (outputs english) are the goal
hebrew_model = Model(hebrew_inputs, hebrew_states)

english_state_input_hidden = Input(shape=(LATENT_DIM,))
english_state_input_cell = Input(shape=(LATENT_DIM,))
english_states_inputs = [english_state_input_hidden, english_state_input_cell]
english_outputs, hidden_state, cell_state = english_lstm(
    english_inputs, initial_state=english_states_inputs)
english_states = [hidden_state, cell_state]
english_outputs = english_dense(english_outputs)
english_model = Model(
    [english_inputs] + english_states_inputs,
    [english_outputs] + english_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, token) for token, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, token) for token, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = hebrew_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_english_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = english_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_token == '\n' or
           len(decoded_sentence) > max_english_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_english_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(8):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = hebrew_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', ' '.join(input_texts[seq_index]))
    print('Decoded sentence:', decoded_sentence)

