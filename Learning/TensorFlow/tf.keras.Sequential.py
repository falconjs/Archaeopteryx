"""
Class Sequential

Inherits From: Model
Aliases:

    Class tf.keras.Sequential
    Class tf.keras.models.Sequential

Defined in tensorflow/python/keras/_impl/keras/engine/sequential.py.

Linear stack of layers.
Arguments:

    layers: list of layers to add to the model.
"""

# Example:
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# Optionally, the first layer can receive an `input_shape` argument:
model = Sequential()
model.add(Dense(32, input_shape=(500,)))

# Afterwards, we do automatic shape inference:
model.add(Dense(32))

# This is identical to the following:
model = Sequential()
model.add(Dense(32, input_dim=500))

# And to the following:
model = Sequential()
model.add(Dense(32, batch_input_shape=(None, 500)))

# Note that you can also omit the `input_shape` argument:
# In that case the model gets built the first time you call `fit` (or other
# training and evaluation methods).
model = Sequential()
model.add(Dense(32))
model.add(Dense(32))
model.compile(optimizer=optimizer, loss=loss)
# This builds the model for the first time:
model.fit(x, y, batch_size=32, epochs=10)

# Note that when using this delayed-build pattern (no input shape specified),
# the model doesn't have any weights until the first call
# to a training/evaluation method (since it isn't yet built):
model = Sequential()
model.add(Dense(32))
model.add(Dense(32))
model.weights  # returns []

# Whereas if you specify the input shape, the model gets built continuously
# as you are adding layers:
model = Sequential()
model.add(Dense(32, input_shape=(500,)))
model.add(Dense(32))
model.weights  # returns list of length 4

# When using the delayed-build pattern (no input shape specified), you can
# choose to manually build your model by calling `build(batch_input_shape)`:
model = Sequential()
model.add(Dense(32))
model.add(Dense(32))
model.build((None, 500))
model.weights  # returns list of length 4
# [<tf.Variable 'dense/kernel:0' shape=(500, 32) dtype=float32_ref>,
#  <tf.Variable 'dense/bias:0' shape=(32,) dtype=float32_ref>,
#  <tf.Variable 'dense_1/kernel:0' shape=(32, 32) dtype=float32_ref>,
#  <tf.Variable 'dense_1/bias:0' shape=(32,) dtype=float32_ref>]