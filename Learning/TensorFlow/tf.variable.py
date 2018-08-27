"""
https://www.tensorflow.org/api_guides/python/state_ops

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
