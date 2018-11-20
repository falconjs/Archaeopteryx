"""
https://www.tensorflow.org/api_docs/python/tf/Variable

https://www.tensorflow.org/guide/variables

"""

import tensorflow as tf

with tf.variable_scope("one"):
    a = tf.get_variable("v", [1])  # a.name == "one/v:0"
with tf.variable_scope("one"):
    b = tf.get_variable("v", [1])  # ValueError: Variable one/v already exists
with tf.variable_scope("one", reuse=True):
    c = tf.get_variable("v", [1])  # c.name == "one/v:0"

with tf.variable_scope("two"):
    d = tf.get_variable("v", [1])  # d.name == "two/v:0"
    e = tf.Variable(1, name="v", expected_shape=[1])  # e.name == "two/v_1:0"

assert(a is c)  # Assertion is true, they refer to the same object.
assert(a is d)  # AssertionError: they are different objects
assert(d is e)  # AssertionError: they are different objects


def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


input1 = tf.random_normal([1, 10, 10, 32])
input2 = tf.random_normal([1, 20, 20, 32])
x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape=[32])  # This fails.
# ValueError: Variable weights already exists, disallowed. Did you mean to set reuse=True or
# reuse=tf.AUTO_REUSE in VarScope? Originally defined at:


def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])


with tf.variable_scope("model"):
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)

# Since depending on exact string names of scopes can feel dangerous, it's also possible to initialize
# a variable scope based on another one:
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
with tf.variable_scope(scope, reuse=True):
  output2 = my_image_filter(input2)


