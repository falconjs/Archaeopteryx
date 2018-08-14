# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '..\\input/notMNIST\\' # Change me to store data elsewhere

"""
====== Problem 1 ======
Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a 
character A through J rendered in a different font. Display a sample of the images that we just downloaded. 
Hint: you can use the package IPython.display.
"""

input_root = "./input/notMNIST/"
notMNIST_large = "notMNIST_large"

# code check for below
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
image_file = input_root + notMNIST_large + "/A/" + "a2F6b28udHRm.png"

display(Image(filename = image_file))
# image_data = imageio.imread(image_file).astype(float)
# 0 -- 255  convert to -0.5 -- 0.5
image_data = (imageio.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth

print(type(image_data))


image_data[0]
image_data[:, 0]