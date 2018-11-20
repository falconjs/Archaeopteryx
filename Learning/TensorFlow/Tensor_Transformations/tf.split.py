"""
https://www.tensorflow.org/api_docs/python/tf/split

Defined in tensorflow/python/ops/array_ops.py.

Splits a tensor into sub tensors.

If num_or_size_splits is an integer type, num_split, then splits value along
dimension axis into num_split smaller tensors. Requires that num_split evenly
divides value.shape[axis].

If num_or_size_splits is not an integer type, it is presumed to
be a Tensor size_splits, then splits value into len(size_splits) pieces.
The shape of the i-th piece has the same size as the value except along
dimension axis where the size is size_splits[i].

"""

import tensorflow as tf
import numpy as np

a = np.array([
    [[[1., 2.], [3., 4.]],
     [[4., 5.], [6., 7.]],
     [[7., 8.], [9., 10.]]],

    [[[1., 2.], [3., 4.]],
     [[4., 5.], [6., 7.]],
     [[7., 8.], [9., 10.]]],

    [[[1., 2.], [3., 4.]],
     [[4., 5.], [6., 7.]],
     [[7., 8.], [9., 10.]]],
]
)

tf_a = tf.constant(a)

# tf_sp0, tf_sp1 = tf.split(tf_a, num_or_size_splits=2, axis=3)
tf_sp0, tf_sp1 = tf.split(tf_a, num_or_size_splits=[1, 2], axis=1)

with tf.Session() as sess:
    print("tf_a:\n", tf_a.eval())
    print("tf_sp0:\n", tf_sp0.eval())
    print("tf_sp1:\n", tf_sp1.eval())



