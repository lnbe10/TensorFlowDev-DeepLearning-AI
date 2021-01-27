import os
import urllib.request as Url
import json

Url.urlretrieve('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json',
filename='sarcasm.json');

with open('sarcasm.json', 'r') as file:
	data = json.load(file);


sentences = [];
labels = [];
urls = [];

for item in data:
	sentences.append(item['headline']);
	labels.append(item['is_sarcastic']);
	urls.append(item['article_link']);


# splitting data into training/test

training_size = 20000;


training_sentences = sentences[0:training_size];
testing_sentences = sentences[training_size:];
training_labels = labels[0:training_size];
testing_labels = labels[training_size:];


# organizing data format to input in NN:

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 1000;
max_len = 120;
trunc_type = 'post';
padding_type = 'post';
oov_tok = '<OOV>';

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>');
tokenizer.fit_on_texts(training_sentences);


word_index = tokenizer.word_index;

training_sequences = tokenizer.texts_to_sequences(training_sentences);
training_padded = pad_sequences(
	training_sequences,
	maxlen=max_len,
	padding=padding_type,
	truncating = trunc_type
	);

testing_sequences = tokenizer.texts_to_sequences(testing_sentences);
testing_padded = pad_sequences(
	testing_sequences,
	maxlen=max_len,
	padding=padding_type,
	truncating = trunc_type
	);

import numpy as np

training_padded = np.array(training_padded);
training_labels = np.array(training_labels);
testing_padded = np.array(testing_padded);
testing_labels = np.array(testing_labels);

# making the model

import tensorflow as tf

embedding_dim = 16;

model = tf.keras.Sequential([
	tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
	tf.keras.layers.Dense(24, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')
	]);

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']);
model.summary();

epochs = 50;
history = model.fit(
	training_padded,
	training_labels,
	epochs=epochs,
	validation_data=(testing_padded,testing_labels),
	verbose=1
	);

model.save('LSTM-sarcasm.h5')

# plot acc and loss for train and validation

import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'accuracy');
plot_graphs(history, 'loss');