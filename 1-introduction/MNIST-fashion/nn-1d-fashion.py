import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# tensorflow has an automated push-load-organize data_set for MNIST
# so in this analysis i'm using the ready-to-play set
# in next works i'll learn to organize and prepare the set
# from the beginning!


mnist = tf.keras.datasets.fashion_mnist;

(training_images, training_labels), (test_images, test_labels) = mnist.load_data();

np.set_printoptions(linewidth=200);


# checking data pieces:
# for i in range(0,9):
# 	print(training_labels[i]);
# 	plt.imshow(training_images[i]);
# 	plt.show();




#normalizing data

training_images = training_images / 255;
test_images = test_images / 255;



#callbacks, so I can stop when a condition is reached

class Callback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if( logs.get('loss') < 0.23 ):
			print('\nTouched 0.23 loss, so cancelling training!\n');
			self.model.stop_training = True;
		if( logs.get('acc') > 0.95 ):
			print('\nTouched 95\% accuracy, so cancelling training!\n');
			self.model.stop_training = True;
callbacks = Callback();

#making nn
"""
  Dimensions
   image        Flatten      Dense         Dense
  (28x28) --->  (784x1)  ---> (128x784)---> (10x128)

  Flatten: just take the image and turns into a line
  (28x28) --> (784x1), so this nn can't take any advantage
  of 2d characteristcs of the images!!!!!!!!!!!!!!!

  Relu: in each neuron, if X > 0 -> output X
  						else     -> output 0
  That's a binary activator!!!

  Softmax: takes the biggest one of the values from the
  layer (n x 1): Used in the last layer for labeling problems
  like this one.

  Dense: fully connected out-neurons from the last layer to the
  entry-neurons from the actual layer 
  

"""

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
								    tf.keras.layers.Dense(128, activation = tf.nn.relu),
								    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
								   ]);


model.compile(optimizer = tf.optimizers.Adam(),
			  loss = 'sparse_categorical_crossentropy',
			  metrics = ['accuracy']
			  );

print('\n\n\nstart training:\n\n\n');
model.fit(training_images, training_labels, epochs = 15, callbacks = [callbacks]);

print('\n\n\nevaluating test:\n\n\n');
model.evaluate(test_images, test_labels);



#storing the test_information

classifications = model.predict(test_images)

#checking the results for test_set:

for i in range(0,20):
	print(classifications[i]);
	print(test_labels[i]);
	plt.imshow(test_images[i]);
	plt.show();