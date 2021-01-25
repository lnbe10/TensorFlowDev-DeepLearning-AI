import os

tr_ho_dir = os.path.join('horse-or-human/horses');
tr_hu_dir = os.path.join('horse-or-human/humans');
vl_ho_dir = os.path.join('validation-horse-or-human/horses');
vl_hu_dir = os.path.join('validation-horse-or-human/humans');

tr_ho_names = os.listdir(tr_ho_dir);
tr_hu_names = os.listdir(tr_hu_dir);
vl_ho_names = os.listdir(vl_ho_dir);
vl_hu_names = os.listdir(vl_hu_dir);


print('total training horse images:', len(tr_ho_names));
print('total training human images:', len(tr_hu_names));
print('total validation horse images:', len(vl_ho_names));
print('total validation human images:', len(vl_hu_names));



import matplotlib.pyplot as plt
import matplotlib.image as mpimg


nrows = 4;
ncols = 4;
pic_index = 0;


fig = plt.gcf();
fig.set_size_inches(ncols * 4, nrows * 4);

pic_index += 8
next_horse_pix = [os.path.join(tr_ho_dir, fname) 
                for fname in tr_ho_names[pic_index-8:pic_index]];
next_human_pix = [os.path.join(tr_hu_dir, fname) 
                for fname in tr_hu_names[pic_index-8:pic_index]];

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  sp = plt.subplot(nrows, ncols, i + 1);
  sp.axis('Off');

  img = mpimg.imread(img_path);
  plt.imshow(img);

plt.show();

import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
]);

model.summary();

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy']);


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        'validation-horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')


class Callback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if( logs.get('loss') < 0.23 ):
			print('\nTouched 0.23 loss, so cancelling training!\n');
			self.model.stop_training = True;
		if( logs.get('accuracy') > 0.85 ):
			print('\nTouched 95\% accuracy, so cancelling training!\n');
			self.model.stop_training = True;

callbacks = Callback();


history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=2,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8,
      callbacks=[callbacks]);


import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
horse_img_files = [os.path.join(tr_ho_dir, f) for f in tr_ho_names]
human_img_files = [os.path.join(tr_hu_dir, f) for f in tr_hu_names]
img_path = random.choice(horse_img_files + human_img_files)

img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

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

plt.show();




