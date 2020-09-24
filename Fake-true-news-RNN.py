import pandas as pd
import numpy as np
from collections import Counter
from keras.preprocessing import sequence
from string import punctuation
from sklearn.model_selection import train_test_split

# reading the dataset
true_dataset = pd.read_csv("True.csv")
fake_dataset = pd.read_csv("Fake.csv")
true_texts = np.array(true_dataset.iloc[:, 1].values, dtype='str')
fake_texts = np.array(fake_dataset.iloc[:, 1].values, dtype='str')

# Decapitalising the texts
true_texts = [text.lower() for text in true_texts]
fake_texts = [text.lower() for text in fake_texts]

# Removing any punctuation
true_all = [text + '\n' for text in true_texts]
fake_all = [text + '\n' for text in fake_texts]

true_all = ''.join([c for t in true_all for c in t if c not in punctuation])
fake_all = ''.join([c for t in fake_all for c in t if c not in punctuation])

# Splitting the texts
true_texts = true_all.split('\n')[:-1]
fake_texts = fake_all.split('\n')[:-1]

# Creating a vocab to int mapping dictionary
true_all_2 = ' '.join(true_texts)
fake_all_2 = ' '.join(fake_texts)
true_words = true_all_2.split()
fake_words = fake_all_2.split()
true_count = Counter(true_words)
fake_count = Counter(fake_words)
len_true = len(true_count)
len_fake = len(fake_count)
sorted_true = true_count.most_common(len_true)
sorted_fake = fake_count.most_common(len_fake)

true_to_int = {w:i+1 for i, (w, f) in enumerate(sorted_true)}
fake_to_int = {w:i+1 for i, (w, f) in enumerate(sorted_fake)}

# Encoding the words
true_encoded = []
for text in true_texts:
    n_text = [true_to_int[word] for word in text.split()]
    true_encoded.append(n_text)

fake_encoded = []
for text in fake_texts:
    n_text = [fake_to_int[word] for word in text.split()]
    fake_encoded.append(n_text)

# Creating the labels
true_encoded_labels = np.ones(len(true_encoded))
fake_encoded_labels = np.zeros(len(fake_encoded))

# Visualizing the lengths of the texts
import matplotlib.pyplot as plt

len_true_2 = [len(x) for x in true_encoded]
pd.Series(len_true_2).hist()
plt.show()

pd.Series(len_true_2).describe()

len_fake_2 = [len(x) for x in fake_encoded]
pd.Series(len_fake_2).hist()
plt.show()

pd.Series(len_fake_2).describe()

# Removing too short texts
true_encoded = [true_encoded[i] for i, l in enumerate(len_true_2) if l > 0]
fake_encoded = [fake_encoded[i] for i, l in enumerate(len_fake_2) if l > 0]

true_encoded_labels = [true_encoded_labels[i] for i, l in enumerate(len_true_2) if l > 0]
fake_encoded_labels = [fake_encoded_labels[i] for i, l in enumerate(len_fake_2) if l > 0]

# Padding and truncating the texts
seq_length = 500
true_encoded = sequence.pad_sequences(true_encoded, maxlen=seq_length)
fake_encoded = sequence.pad_sequences(fake_encoded, maxlen=seq_length)

# Joining the texts and labels
texts = np.concatenate((true_encoded, fake_encoded), axis=0)
labels = np.concatenate((true_encoded_labels, fake_encoded_labels), axis=0)

# Reshaping the text and labels
texts = np.reshape(texts, (texts.shape[0], texts.shape[1], 1))
labels = np.reshape(labels, (labels.shape[0], 1))

# Train test splitting
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Make the model and train it
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences=True, input_shape = (500, 1)))
regressor.add(LSTM(units = 25, return_sequences=True))
regressor.add(LSTM(units = 25))
regressor.add(Dense(units = 1, activation="sigmoid"))

regressor.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

regressor.fit(X_train, y_train, batch_size = 32, epochs = 3)

# Test the model
y_pred = regressor.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

