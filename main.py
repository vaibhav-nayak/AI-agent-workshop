import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, LSTM


data = "He loves to read books. He is going to school. The sun rises in the east. He plays football every evening. They are watching a movie. My dog likes to bark. We went to the park. It is raining today. You should drink water. Birds fly in the sky."

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
word_to_index = tokenizer.word_index
vocab_size = len(word_to_index) + 1

sequences = []
for line in data.split('.'):
    if not line: continue
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in sequences])
sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))

X = sequences[:,:-1]
y = sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)


# model = Sequential([
#     Embedding(input_dim=vocab_size, output_dim=10, input_length=max_sequence_len-1),
#     Flatten(),
#     Dense(100, activation='relu'),
#     Dense(vocab_size, activation='softmax')
# ])

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=10, input_length=max_sequence_len-1),
    LSTM(50),
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=200, verbose=1)


def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]
    predicted_index = np.argmax(predicted_probs)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

input_test = "She"
next_word = predict_next_word(model, tokenizer, input_test, max_sequence_len)

while next_word is not None:
    input_test += " " + next_word
    print(next_word)
    time.sleep(1)
    next_word = next_word + " " + predict_next_word(model, tokenizer, next_word, max_sequence_len)
