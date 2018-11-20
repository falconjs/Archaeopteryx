"""
https://www.tensorflow.org/api_docs/python/tf/shape

Defined in tensorflow/python/ops/array_ops.py.

Returns the shape of a tensor.

This operation returns a 1-D integer tensor representing the shape of input.

"""

import tensorflow as tf

t = tf.constant([[[1, 1, 1],
                  [2, 2, 2]],
                 [[3, 3, 3],
                  [4, 4, 4]]])

tf_shape = tf.shape(t)  # [2, 2, 3]

# init = tf.global_variables_initializer()

with tf.Session() as sess:
    # init.run()
    print(tf_shape.eval())  # [2 2 3]

"""
Case 2 : placeholder need to be fed before check shape.
"""

# tf_train_dataset = tf.placeholder(tf.float32,
#                                   shape=(None, 3, 3, 1))  # [?, 3, 3, 1]

tf_train_dataset = tf.placeholder(tf.float32,
                                  shape=(3, 3, 3) + (1,))  # [3, 3, 3, 1]

# tf_train_dataset = tf.placeholder(tf.float32,
#                                   shape=(None,) + (3, 3, 1)) # [3, 3, 3, 1]

with tf.Session() as sess:
    # init.run()
    print(tf_train_dataset.eval())

# You must feed a value for placeholder tensor 'Placeholder' with dtype float and shape [3,3,3,1]
