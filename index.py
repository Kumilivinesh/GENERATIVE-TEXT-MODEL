import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

text_data = """
Artificial Intelligence (AI) is rapidly transforming the world we live in. From healthcare and finance to transportation and education,
AI is making processes smarter, faster, and more efficient. Machine learning, a branch of AI, enables systems to learn and improve from experience without being explicitly programmed.
Deep learning, a subset of machine learning, utilizes neural networks with many layers to analyze complex patterns in large datasets.
One of the most powerful aspects of AI is its ability to understand natural language, which allows machines to interact with humans in a more intuitive way.
In healthcare, AI algorithms can detect diseases from medical images with remarkable accuracy. In finance, it helps in fraud detection and market analysis. Autonomous vehicles rely on AI to make real-time decisions on the road. Despite its benefits,
AI also poses ethical challenges, such as job displacement and data privacy concerns.
As we move forward, it is important to ensure that AI is developed responsibly and used for the benefit of all humanity.
"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])
total_words = len(tokenizer.word_index) + 1
print("Total Vocabulary:", total_words)

# Generate sequences
input_sequences = []
token_list = tokenizer.texts_to_sequences([text_data])[0]

for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)

max_seq_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_seq_len - 1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X, y, epochs=100, verbose=1)

def generate_text(seed_text, next_words=150):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

print(generate_text("Artificial Intelligence is", next_words=166))
