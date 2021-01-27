# Rock-Paper-Scissors hand identifier
# A multiclass classifier (with 3 classes .-.)

import os


# 1 - organizing data
# got it from laurencemoroney.com, thanks!

rock_dir = os.path.join('rps/rock');
rock_lst = os.listdir(rock_dir);
print('total training rock images: ', len(rock_lst));

paper_dir = os.path.join('rps/rock');
paper_lst = os.listdir(paper_dir);
print('total training paper images: ', len(rock_lst));

scissors_dir = os.path.join('rps/scissors');
scissors_lst = os.listdir(scissors_dir);
print('total training scissors images: ', len(rock_lst));

# 2- showing data

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np

pic_num = 0;

rock_random = random.sample(rock_lst, len(rock_lst));
paper_random = random.sample(paper_lst, len(paper_lst));
scissors_random = random.sample(scissors_lst, len(scissors_lst));

next_rock = [os.path.join(rock_dir, file)
	for file in rock_random[:pic_num]];

next_paper = [os.path.join(paper_dir, file)
	for file in paper_random[:pic_num]];

next_scissors = [os.path.join(scissors_dir, file)
	for file in scissors_random[:pic_num]];


for img_path in next_rock+next_paper+next_scissors:
	img = mpimg.imread(img_path);
	plt.imshow(img);
	plt.axis('Off');
	plt.show(); 

# 3 - making CNN model

import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

tr_d = 'rps';
tr_datagen = ImageDataGenerator(
	rescale = 1./255.,
	rotation_range = 40,
	width_shift_range = 0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest'
	);
tr_gen = tr_datagen.flow_from_directory(
	tr_d,
	target_size=(150,150),
	class_mode='categorical',
	batch_size=30
	);

vl_d = 'rps-test-set';
vl_datagen = ImageDataGenerator(rescale=1./255.);
vl_gen = vl_datagen.flow_from_directory(
	vl_d,
	target_size=(150,150),
	class_mode='categorical',
	batch_size=30
	);

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
]);

model.summary();

model.compile(
	loss='categorical_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy']
	);

history = model.fit(
	tr_gen, epochs=20, steps_per_epoch=10,
	validation_data = vl_gen, verbose=1,
	validation_steps=10);

# first time saving the CNN after training!

model.save('rps.h5');

# plotting results (loss,acc):

acc = history.history['accuracy'];
val_acc = history.history['val_accuracy'];
loss = history.history['loss'];
val_loss = history.history['val_loss'];

epochs = range(len(acc));

plt.plot(epochs, acc, 'r', label='Training accuracy');
plt.plot(epochs, val_acc, 'b', label='Validation accuracy');
plt.title('Training and validation accuracy');
plt.legend(loc=0);

plt.show();

# predicting from files:

new_files_path = os.path.join('rps_new');
new_lst = os.listdir(new_files_path);

for file in new_lst:
	fpath = os.path.join(new_files_path,file);

	img = image.load_img(fpath, target_size=(150,150));
	x = image.img_to_array(img);
	x = np.expand_dims(x, axis=0);
	images = np.vstack([x]);
	classes = model.predict(images, batch_size=10);
	class_def = np.round_(classes);

	if class_def[0][0] == 1.:
		print(file, ' is a paper');
	if class_def[0][1] == 1.:
		print(file, ' is a rock');
	if class_def[0][2] == 1.:
		print(file, ' is a scissor');
