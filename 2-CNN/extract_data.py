import os
import zipfile
import urllib.request as url




url.urlretrieve('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip',
	filename='rps.zip'
	);
url.urlretrieve('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip',
	filename='rps-test-set.zip'
	);
url.urlretrieve('https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip',
	filename='cats-and-dogs-filtered.zip'
	);

local_zip = ['rps.zip', 'rps-test-set.zip','cats-and-dogs-filtered.zip'];

for file in local_zip:
	zip_ref = zipfile.ZipFile(file, 'r');
	zip_ref.extractall('');
	zip_ref.close();

