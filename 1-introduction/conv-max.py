# I'm learning about how good
# convolutions can make some characteristcs
# in an image clearer, so, easily to classify
# Did a function to Maxpooling images too :D

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc



i = misc.ascent();
size_x = i.shape[0];
size_y = i.shape[1];


# showing image

print("\nshowing image\n");
plt.grid(False);
plt.gray();
plt.axis('off');
plt.imshow(i);
plt.show();


# conv filters should sum up elements to 0 or 1, so:
# vert_sum = -1-2-1+0+0+0+1+2+1 = 0

vert_filter 	= [[-1,-2,-1], [0,0,0], [1,2,1]];
hor_filter  	= [[-1, 0, 1], [-2,0,2], [-1,0,1]];
contour_filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]];

weight = 1;


def Convolution(filter, i, weight, show):
	"""
		in the image pixels, we have:
		the iter starts in (x,y) = (1,1)
		the iter ends in (x,y) = (size_x-1, size_y-1)
		so the border is out of the iteration!!
		in the example above:
		x = element out of iter
		o = element in iter

				 1 2 3 ...... size_x-1  size_x
		1		 x x x  x x x    x        x
		2		 x o o  o o o    o        x
		3		 x o o  o o o    o        x
		.		 x o o  o o o    o        x
		.		 x o o  o o o    o        x
		.		 x o o  o o o    o        x
		size_y-1 x o o  o o o    o        x
		size_y   x x x  x x x    x        x 


		in each iteration, we will apply a element_wise_matrix_mult
		over the element i and its surrounding with the convolution matrix:

		A^.*B = [a11*b11  a12*b22 ..... ]
				[a21*b21  a22*b22 ..... ]
		
		And then we sum all the elements of this matrix...
		so, the pixel addresses for the surrounding are:		
		(x-1, y-1)  (x  , y-1)  (x+1, y-1)
		(x-1, y  )  (x  , y  )  (x+1, y  )
		(x-1, y+1)  (x  , y+1)  (x+1, y+1)
	"""


	size_x = i.shape[0];
	size_y = i.shape[1];
	i_transformed = np.copy(i);
	for x in range(1,size_x-1):
		for y in range(1,size_y-1):
			convolution  = 0.0;
			convolution += (i[x-1, y-1] * filter[0][0])
			convolution += (i[x  , y-1] * filter[0][1])
			convolution += (i[x+1, y-1] * filter[0][2])
			convolution += (i[x-1, y  ] * filter[1][0])
			convolution += (i[x  , y  ] * filter[1][1])
			convolution += (i[x+1, y  ] * filter[1][2])
			convolution += (i[x-1, y+1] * filter[2][0])
			convolution += (i[x  , y+1] * filter[2][1])
			convolution += (i[x+1, y+1] * filter[2][2])
			convolution  = convolution * weight
		
			# now we have to guarantee that the sum of the
			# element is in (0,255), so its a legal value:
			if(convolution<0):
				convolution = 0;
			if(convolution>255):
				convolution = 255;
			i_transformed[x,y] = convolution;

	# printing image if user requires
	if(show == True):
		plt.gray();
		plt.grid(False);
		plt.axis('off');
		plt.imshow(i_transformed);
		plt.show();

	return i_transformed;


def Maxpool(i, step, show):
	"""
		Maxpool is about to take the higher value in a little set
		So we can take the 'stronger' characteristics of each part
		of an image...
	"""

	size_x = i.shape[0];
	size_y = i.shape[1];

	(max_step_x, x_ok) = divmod(size_x, step);
	(max_step_y, y_ok) = divmod(size_y, step);

	if(x_ok == 0 and y_ok == 0):
		i_pool = np.zeros((max_step_x, max_step_y));
		for x in range(0,max_step_x-1):
			for y in range(0,max_step_y-1):
				pixel_pool = i[x*step:x*step+step-1,y*step:y*step+step-1].max();
				i_pool[x,y] = pixel_pool;
	if (x_ok != 0 and y_ok == 0):
		print("the ", step, " steps cannot divide the x dimension ", size_x);
		return;
	if (x_ok == 0 and y_ok != 0):
		print("the ", step, " steps cannot divide the x dimension ", size_y);
		return;
	if (x_ok != 0 and y_ok != 0):
		print("the ", step, " steps cannot divide the x dimension ", size_x, " and the y dimension ", size_y);
		return;

	# printing image if user requires
	if(show == True):
		print("pooling with ", step, " pixel steps");
		plt.gray();
		plt.grid(False);
		plt.axis('off');
		plt.imshow(i_pool);
		plt.show();


	return i_pool


# that's the result applying some simple conv filters in an image
# in grayscale....


i_contour	= Convolution(contour_filter, i, weight, True);
i_vert		= Convolution(vert_filter,	  i, weight, True);
i_hor		= Convolution(hor_filter,	  i, weight, True);

i_maxpool_2 = Maxpool(i,  2,  True);
i_maxpool_4 = Maxpool(i,  4,  True);
i_maxpool_8 = Maxpool(i,  8,  True);
i_maxpool_16= Maxpool(i, 16,  True);

i_maxpool_17= Maxpool(i[0:512,0:17], 17,  True);