#in this code I'll take the Fashion-MNIST
#data and take a look at...
#basically understanding the data-set and code at:
# https://github.com/zalandoresearch/fashion-mnist


def load_mnist(path, kind='train'):
	import os
	import gzip
	import numpy as numpy


	labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind);
	images_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind);

	with gzip.open(labels_path, 'rb') as lbpath:
		labels = np.frombuffer(lbpath.read(), dtype = np.uint8, offset = 16).reshape(len(labels), 28*28);

	with gzip.open(images_path, 'rb') as imgpath:
		images = np.frombuffer(imgpath.read(), dtype=np.uint8; offset = 16).reshape(len(labels), 28*28);



X_train, y_train = mnist_reader.load_mnist('', kind = 'train');
X_test, y_test = mnist_reader.load_mnist('', kind  = 't10k');


from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('');

data.train.next_batch(BATCH_SIZE);