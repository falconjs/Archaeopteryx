"""
https://www.tensorflow.org/api_docs/python/tf/math/

Defined in tensorflow/_api/v1/math/__init__.py.

Basic arithmetic operators.

See the python/math_ops guide.

"""

import tensorflow as tf
import numpy as np

a = np.array([
    [[[1., 2.], [3., 4.]],
     [[4., 5.], [6., 7.]],
     [[7., 8.], [9., 10.]]],

    [[[1., 2.], [3., 4.]],
     [[4., 5.], [6., 7.]],
     [[7., 8.], [9., 10.]]]
]
)

tf_square = tf.square(a)
# [[[[  1.   4.]   [  9.  16.]]
#   [[ 16.  25.]   [ 36.  49.]]
#   [[ 49.  64.]   [ 81. 100.]]]
#  [[[  1.   4.]   [  9.  16.]]
#   [[ 16.  25.]   [ 36.  49.]]
#   [[ 49.  64.]   [ 81. 100.]]]

tf_reduce_sum = tf.reduce_sum(tf_square, 1, keepdims=True)

with tf.Session() as sess:
    print("tf_square:\n", tf_square.eval())
    print("tf_reduce_sum:\n", tf_reduce_sum.eval())


