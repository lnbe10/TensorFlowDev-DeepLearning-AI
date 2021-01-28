# it's important to check a little bit of
# how we can build some models of time series
# before analyse some real data in this category..
# also, got some ploting skills..

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_series(time, series, format='-', start=0, end=None, label=None):
	plt.plot(time[start:end], series[start:end], format, label=label);
	plt.xlabel('Time');
	plt.ylabel('Value');
	if label:
		plt.legend(fontsize=14);
	plt.grid(True);


# takes a timeline, multiply by a 'slope'
# if it's a straight line, we'll get a line equation .-.
def trend(time, slope=0):
	return slope*time;


time = np.arange(4*365+1);
# time = [0, 1, 2, ......, 1399, 1400]
baseline = 10;
series = trend(time, 0.1);
#series = line with slope = 0.1 .-.

plt.figure(figsize=(10,6));
plot_series(time, series, label='simple line');



def seasonal_pattern(season_time):
	return np.where(season_time < 0.4,
		np.cos(season_time * 2* np.pi),
		1 / np.exp(3 * season_time)
		);


def seasonality(time, period, amplitude=1, phase=0):
	season_time = ((time+phase) % period) / period;
	return amplitude * seasonal_pattern(season_time);


baseline = 10;
amplitude = 40;
series2 = seasonality(time, period=365, amplitude=amplitude);

plt.figure(figsize=(10,6))
plot_series(time, series2, label='pure seasonality');



slope = 0.05
series3 = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series3, label='line+seasonality');



def white_noise(time, noise_level=1, seed=None):
	rnd = np.random.RandomState(seed);
	return rnd.randn(len(time))*noise_level;


noise_level = 5;
noise = white_noise(time, noise_level, seed=42);

plt.figure(figsize=(10, 6));
plot_series(time, noise, label='pure noise');


series4 = series3 + noise;

plt.figure(figsize=(10, 6))
plot_series(time, series4, label='line+season+noise');



def autocorrelation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    φ = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += φ * ar[step - 1]
    return ar[1:] * amplitude

#series5 = autocorrelation(time, 10, seed=42) + trend(time, 0.1);
#plot_series(time, series5, 'g')

def moving_average(series, window_size):
	forecast=[];
	time_avg=[];
	if window_size==1:
		return series
	else:
		for time in range(window_size, len(series)):
			mean = series[time:time+window_size].mean();
			forecast.append(mean);
			time_avg.append(time);
		return np.array(forecast), np.array(time_avg);

moving_avg, time_avg = moving_average(series4, 30);

plt.figure(figsize=(10, 6))
plot_series(time, series4, 'r');
plot_series(time_avg, moving_avg,'b', label='moving_avg')
plt.show()

