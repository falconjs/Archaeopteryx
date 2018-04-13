# ==============================================================================
"""
The MNIST data is hosted on Yann LeCun's website.
If you are copying and pasting in the code from this tutorial,
start here with these two lines of code which will download and read in the data automatically:
"""
# ==============================================================================

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

