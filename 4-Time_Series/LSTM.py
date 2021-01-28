import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level;


time = np.arange(365*4+1, dtype='float32');
baseline = 10;
amplitude = 40;
slope = 0.05;
noise_level = 5;

series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude);
series += noise(time, noise_level, seed=42);

#split_time = st
st = 1000;
time_train = time[:st];
x_train = series[:st];
time_valid = time[st:];
x_valid = series[st:];

window_size = 20;
batch_size = 32;
shuffle_buffer_size = 1000;

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series);
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True);
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1));
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]));
  dataset = dataset.batch(batch_size).prefetch(1);
  return dataset;



# running model to get best learning rate

tf.keras.backend.clear_session();
tf.random.set_seed(51);
np.random.seed(51);

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size);

model = tf.keras.models.Sequential([
	tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
	tf.keras.layers.Dense(1),
	tf.keras.layers.Lambda(lambda x: x * 100.0)
]);

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20));

optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9);

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"]
              );

model.summary();
history = model.fit(dataset, epochs=100, callbacks=[lr_schedule]);

# looking the loss function to determine lr:

plt.semilogx(history.history["lr"], history.history["loss"]);
plt.axis([1e-8, 1e-3, 0, 30]);
plt.show();


# running again with adjusted learning rate

tf.keras.backend.clear_session();
tf.random.set_seed(51);
np.random.seed(51);

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size);

model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
]);

model.summary();
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),metrics=["mae"]);

history = model.fit(dataset,epochs=500,verbose=0);


forecast = [];
results = [];

for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]));

forecast = forecast[st-window_size:];
results = np.array(forecast)[:, 0, 0];


plt.figure(figsize=(10, 6));

plot_series(time_valid, x_valid, label='Data');
plot_series(time_valid, results, label='Prediction');
plt.show();


tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()






import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

mae=history.history['mae']
loss=history.history['loss']

epochs=range(len(loss))


plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])
plt.show()


plt.figure()

epochs_zoom = epochs[200:]
mae_zoom = mae[200:]
loss_zoom = loss[200:]


plt.plot(epochs_zoom, mae_zoom, 'r')
plt.plot(epochs_zoom, loss_zoom, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])
plt.show()





# tf.keras.backend.clear_session()
# dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
#                       input_shape=[None]),
#   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
#   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#   tf.keras.layers.Dense(1),
#   tf.keras.layers.Lambda(lambda x: x * 100.0)
# ])

# model.summary();
# model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
# model.fit(dataset,epochs=100, verbose=0);
