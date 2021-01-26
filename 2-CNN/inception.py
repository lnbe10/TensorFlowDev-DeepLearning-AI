import os

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5';

pre_trained_model = InceptionV3(
	input_shape = (150,150,3),
	include_top = False,
	weights = None
	);

pre_trained_model.load_weights(weights_file);

for layer in pre_trained_model.layers:
	layer.trainable = False;

last_layer = pre_trained_model.get_layer('mixed7');
print('last layer output shape: ', last_layer.output_shape);
last_output = last_layer.output;

from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(last_output);
x = layers.Dense(1024, activation = 'relu')(x);
x = layers.Dropout(0.2)(x);
x = layers.Dense(1, activation = 'sigmoid')(x);

model = Model(pre_trained_model.input, x);

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy']
              );

#model ready, pre-processing inputs:

from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
	tr_dir, batch_size=5,
	class_mode='binary', target_size=(150,150)
	);

vl_generator = validation_datagen.flow_from_directory(
	vl_dir, batch_size=5,
	class_mode='binary', target_size=(150,150)
	);

# history saving and run

history = model.fit(
            tr_generator,
            validation_data = vl_generator,
            steps_per_epoch = 100,
            epochs = 20,
            validation_steps = 50,
            verbose = 2
            );

# plot loss and acc

import matplotlib.pyplot as plt

acc = history.history['accuracy'];
val_acc = history.history['val_accuracy'];
loss = history.history['loss'];
val_loss = history.history['val_loss'];

epochs = range(len(acc));

plt.plot(epochs, acc, 'r', label='Training accuracy');
plt.plot(epochs, val_acc, 'b', label='Validation accuracy');
plt.title('Training and validation accuracy');
plt.legend(loc=0);
plt.figure();

plt.plot(epochs, loss, 'r', label='Training loss');
plt.plot(epochs, val_loss, 'b', label='Validation loss');
plt.title('Training and validation loss');
plt.legend();
plt.figure();

plt.show();