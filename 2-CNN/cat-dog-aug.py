## CNN that ends in a sigmoid to predict
## if a image is a dog or cat
## 1- here we take the data, prepare it,
##    use the built-in tensorflow method
##    to label data and do some augmentation
## 2- then we run the model, get the loss/acc
##    and other stats from history ploted
## 3- Next we put new data to the CNN predict
##    and see the results
## 4- Finally, we save all the Layers' output
##    for a random input image, and plot them
##    so we can have an idea of what the
##    convolutional filters are doing!!
## 5- That's ******* amazing!

import os

mdir = 'cats_and_dogs_filtered';

tr_dir = os.path.join(mdir, 'train');
vl_dir = os.path.join(mdir, 'validation');

dog_td = os.path.join(tr_dir, 'dogs');
cat_td = os.path.join(tr_dir, 'cats');

dog_vd = os.path.join(vl_dir, 'dogs');
cat_vd = os.path.join(vl_dir, 'cats');

dog_td_fnames = os.listdir(dog_td);
cat_td_fnames = os.listdir(cat_td);

dog_vd_fnames = os.listdir(dog_vd);
cat_vd_fnames = os.listdir(cat_vd);

print('total training cat images:', len(cat_td_fnames));
print('total training dog images:', len(dog_td_fnames));
print('total validation cat images:', len(cat_vd_fnames));
print('total validation dog images:', len(dog_vd_fnames));


import tensorflow as tf

model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (150,150,3)),
	#out size (148,148, 3)
	tf.keras.layers.MaxPooling2D(2,2),
	#out size ( 74, 74, 3)
	tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
	#out size ( 72, 72, 3)
	tf.keras.layers.MaxPooling2D(2,2),
	#out size ( 36, 36, 3)
	tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
	#out size ( 34, 34, 3)
	tf.keras.layers.MaxPooling2D(2,2),
	#out size ( 17, 17, 3)
	#I don't think flatten the data in this stage will help
	#bcs we get a very big 1xN matrix.....
	#and lose all 2d shapes info we could still work with...
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, activation = 'relu'),
	tf.keras.layers.Dense(1, activation = 'sigmoid')
	]);

model.summary();


from tensorflow.keras.optimizers import RMSprop

model.compile(
	optimizer = RMSprop(lr = 0.001),
	loss = 'binary_crossentropy',
	metrics = ['accuracy'],
	)



# pre-processing data -> 150,150,3:
# data augmentation added -> some rotations,
# flips and zoom in images as they're feeded
# into the CNN

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
	rescale = 1./255.,
	rotation_range = 40,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	shear_range = 0.2,
	zoom_range = 0.2,
	horizontal_flip = True,
	fill_mode = 'nearest'
	);


validation_datagen = ImageDataGenerator( rescale = 1./255.);

tr_generator = train_datagen.flow_from_directory(
	tr_dir, batch_size=20,
	class_mode='binary', target_size=(150,150)
	);

vl_generator = validation_datagen.flow_from_directory(
	vl_dir, batch_size=20,
	class_mode='binary', target_size=(150,150)
	);

history = model.fit(
	tr_generator,
	steps_per_epoch=100,
	epochs=2,
	verbose=2,
	validation_data = vl_generator,
	validation_steps = 100,
	);


#plotting from history

import matplotlib.pyplot as plt

acc = history.history['accuracy'];
val_acc = history.history['val_accuracy'];
loss = history.history['loss'];
val_loss = history.history['val_loss'];

epochs = range(len(acc));

plt.plot(epochs, acc, 'r', label='Training accuracy');
plt.plot(epochs, val_acc, 'b', labe='Validation accuracy');
plt.title('Training and validation accuracy');

plt.figure();

plt.plot(epochs, loss, 'r', label='Training loss');
plt.plot(epochs, val_loss, 'b', labe='Validation loss');
plt.title('Training and validation loss');
plt.legend();

plt.show();



#predicting from new data:

import numpy as np
from keras.preprocessing import image

newdir = 'new_pics';
newdir_fnames = os.listdir(newdir);

def process_image(path):
	img = image.load_img(path, target_size=(150,150));
	x = image.img_to_array(img);
	return x;	



for file in newdir_fnames:
	path = os.path.join(newdir, file);
	img = image.load_img(path, target_size=(150,150));
	x = image.img_to_array(img);
	x = np.expand_dims(x, axis=0);

	images = np.vstack([x]);

	classification = model.predict(images, batch_size = 10);

	if classification[0]>0:
		print(file + 'is a dog');
	else:
		print(file + 'is a cat');


# looking the Conv Filters process:

import matplotlib.image as mpimg
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

sucessive_outputs = [layer.output for layer in model.layers[1:]];
visualization_model = tf.keras.models.Model(
	inputs = model.input,
	outputs = sucessive_outputs
	);


cat_path_archive = [os.path.join(cat_td, f) for f in cat_td_fnames];
dog_path_archive = [os.path.join(dog_td, f) for f in dog_td_fnames];


img_path = random.choice(cat_path_archive+dog_path_archive);

img = image.load_img(img_path, target_size=(150,150));

x = image.img_to_array(img);
x = x.reshape((1,)+x.shape);
x /= 255.;

sucessive_feature_maps = visualization_model.predict(x);

layer_names = [layer.name for layer in model.layers]

for layer_name, feature_map in zip(layer_names, sucessive_feature_maps):

	if len(feature_map.shape) == 4:
		n_features	= feature_map.shape[-1];
		size		= feature_map.shape[ 1];

		display_grid = np.zeros((size, size*n_features));

		for i in range(n_features):
			x = feature_map[0,:,:,i];
			x -= x.mean();
			x /= x.std();
			x *=  64;
			x += 128;
			x = np.clip(x,0,255).astype('uint8');
			display_grid[:,i*size:(i+1)*size] = x;

		scale = 20./n_features;
		plt.figure(figsize = (scale*n_features, scale));
		plt.title(layer_name);
		plt.grid(False);
		plt.imshow( display_grid, aspect='auto', cmap='viridis');


