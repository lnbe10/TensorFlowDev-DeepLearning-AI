import numpy as numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras import models as mds

# tr -> training
# ts -> test
# im -> image
# lb -> label

mnist = tf.keras.datasets.fashion_mnist;
(tr_im, tr_lb), (ts_im, ts_lb) = mnist.load_data();
#normalization
tr_im = tr_im.reshape(60000, 28, 28, 1);
tr_im = tr_im/255;
ts_im = ts_im.reshape(10000, 28, 28, 1);
ts_im = ts_im/255;


# class Callback(tf.keras.callbacks.Callback):
# 	def on_epoch_end(self, epoch, logs={}):
# 		if( logs.get('acc') > 0.998 ):
# 			print('\nTouched 99.8\% accuracy, so cancelling training!\n');
# 			self.model.stop_training = True;

# callbacks = Callback();    



model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')	
	]);

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']);
model.summary();
#model.fit(tr_im, tr_lb, epochs=15, callbacks = [callbacks]);
model.fit(tr_im, tr_lb, epochs=1);
test_loss = model.evaluate(ts_im, ts_lb);



#problems with subplots .-.
# print(ts_lb[:100]);


# layer_out = [layer.output for layer in model.layers];
# activation_model = mds.Model(inputs = model.input, outputs = layer_out);

# f, axarr = plt.subplots(1,4)
# CONVOLUTION_NUMBER = 1;
# shoes = [0, 23, 28];
# for a in range(0,len(shoes)):
# 	s = shoes[a];
# 	for x in range(0,4):
# 		f1 = activation_model.predict(ts_im[s].reshape(1, 28, 28, 1))[x]

# 		axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
# 		axarr[0,x].grid(False)