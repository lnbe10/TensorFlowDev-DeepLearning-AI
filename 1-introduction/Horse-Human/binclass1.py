# binary classifier for human or horse

import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# to unzip the folders....

# lzip = 'horse-or-human.zip';
# zip_ref = zipfile.ZipFile(lzip, 'r');
# zip_ref.extractall('horse-or-human');
# zip_ref.close();

tr_ho_dir = os.path.join('horse-or-human/horses');
tr_hu_dir = os.path.join('horse-or-human/humans');

tr_ho_names = os.listdir(tr_ho_dir);
print(tr_ho_names[:10]);

tr_hu_names = os.listdir(tr_hu_dir);
print(tr_hu_names[:10]);


print('\ntotal training horse images:', len(os.listdir(tr_ho_dir)), '\n');
print('\ntotal training human images:', len(os.listdir(tr_hu_dir)), '\n');


#plot examples of data

nrows = 4;
ncols = 4;
index = 0;

fig = plt.gcf();
fig.set_size_inches(ncols*4,nrows*4);


index += 154;

next_ho = [os.path.join(tr_ho_dir, fname)
			for fname in tr_ho_names[index-8:index]];
next_hu = [os.path.join(tr_hu_dir, fname)
			for fname in tr_hu_names[index-8:index]];

for i, img_path in enumerate(next_ho+next_hu):
	#subplot indices starting at 1
	sp = plt.subplot(nrows, ncols, i+1);
	sp.axis('Off');

	img = mpimg.imread(img_path);
	plt.imshow(img);

plt.show()



# NN

model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')
	]);

model.summary();


model.compile(
	loss='binary_crossentropy',
	optimizer=RMSprop(lr=0.001),
	metrics=['accuracy']
	);


train_datagen = ImageDataGenerator(rescale=1/255);
train_generator = train_datagen.flow_from_directory(
						'horse-or-human',
						target_size = (300,300),
						batch_size = 128,
						class_mode = 'binary'
						);


#training

class Callback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if( logs.get('loss') < 0.23 ):
			print('\nTouched 0.23 loss, so cancelling training!\n');
			self.model.stop_training = True;
		if( logs.get('acc') > 0.85 ):
			print('\nTouched 95\% accuracy, so cancelling training!\n');
			self.model.stop_training = True;
callbacks = Callback();


history = model.fit(
	train_generator,
	steps_per_epoch = 8,
	epochs = 10,
	verbose = 1,
	callbacks = [callbacks]
	);

#seen data


import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

suc_out = [layer.output for layer in model.layers[1:]];
vis_model = tf.keras.models.Model(inputs = model.input, outputs = suc_out);

ho_img_f = [os.path.join(tr_ho_dir, f) for f in tr_ho_names];
hu_img_f = [os.path.join(tr_hu_dir, f) for f in tr_hu_names];
img_path = random.choice(ho_img_f+hu_img_f);

img = load_img(img_path, target_size=(300,300))
x = img_to_array(img);
x = x.reshape((1,) + x.shape);
x /= 255;

successive_feature_maps = vis_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers[1:]]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')