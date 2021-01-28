import csv

import urllib.request as Url

Url.urlretrieve('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv', filename='daily-min-temperatures.csv');


time_step = [];
temps = [];

with open('daily-min-temperatures.csv') as file:
  reader = csv.reader(file);
  next(reader);
  i = 0;
  for line in reader:
    temps.append(line[1]);
    time_step.append(i);
    i += 1;
