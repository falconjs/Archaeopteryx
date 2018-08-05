"""
# Importing TensorFlow

The canonical import statement for TensorFlow programs is as follows:
"""

import tensorflow as tf

"""
This gives Python access to all of TensorFlow's classes, methods, and symbols. Most of the documentation assumes you 
have already done this.

# The Computational Graph

You might think of TensorFlow Core programs as consisting of two discrete sections:

   1. Building the computational graph.
   2. Running the computational graph.

A computational graph is a series of TensorFlow operations arranged into a graph of nodes. Let's build a simple 
computational graph. Each node takes zero or more tensors as inputs and produces a tensor as an output. 
One type of node is a constant. Like all TensorFlow constants, it takes no inputs, and it outputs a value it 
stores internally. We can create two floating point Tensors node1 and node2 as follows:

"""

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

"""
Notice that printing the nodes does not output the values 3.0 and 4.0 as you might expect. 
Instead, they are nodes that, when evaluated, would produce 3.0 and 4.0, respectively. 
To actually evaluate the nodes, we must run the computational graph within a session. 
A session encapsulates the control and state of the TensorFlow runtime.

The following code creates a Session object and then invokes its run method to run enough of the computational graph
to evaluate node1 and node2. By running the computational graph in a session as follows:
"""


sess = tf.Session()
print(sess.run([node1, node2]))

"""
We can build more complicated computations by combining Tensor nodes with operations (Operations are also nodes). 
For example, we can add our two constant nodes and produce a new graph as follows:
"""

from __future__ import print_function
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

"""
TensorFlow provides a utility called TensorBoard that can display a picture of the computational graph. 
Here is a screenshot showing how TensorBoard visualizes the graph:

A graph can be parameterized to accept external inputs, known as placeholders. 
A placeholder is a promise to provide a value later.
"""

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

"""
The preceding three lines are a bit like a function or a lambda in which we define two input parameters (a and b) 
and then an operation on them. We can evaluate this graph with multiple inputs by using the feed_dict argument to 
the run method to feed concrete values to the placeholders:
"""

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

"""
We can make the computational graph more complex by adding another operation. For example,
"""
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

"""
In machine learning we will typically want a model that can take arbitrary inputs, such as the one above. 
To make the model trainable, we need to be able to modify the graph to get new outputs with the same input. 
Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value:
"""

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

"""
Constants are initialized when you call tf.constant, and their value can never change. By contrast, 
variables are not initialized when you call tf.Variable. To initialize all the variables in a TensorFlow program, 
you must explicitly call a special operation as follows:
"""

init = tf.global_variables_initializer()
sess.run(init)

"""
It is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables. 
Until we call sess.run, the variables are uninitialized.

Since x is a placeholder, we can evaluate linear_model for several values of x simultaneously as follows:
"""

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
print(sess.run([W, b]))


"""
We've created a model, but we don't know how good it is yet. To evaluate the model on training data, 
we need a y placeholder to provide the desired values, and we need to write a loss function.

A loss function measures how far apart the current model is from the provided data. We'll use a standard loss model 
for linear regression, which sums the squares of the deltas between the current model and the provided data. 
linear_model - y creates a vector where each element is the corresponding example's error delta. 
We call tf.square to square that error. Then, we sum all the squared errors to create a single scalar 
that abstracts the error of all examples using tf.reduce_sum:
"""

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

"""
We could improve this manually by reassigning the values of W and b to the perfect values of -1 and 1. 
A variable is initialized to the value provided to tf.Variable but can be changed using operations like tf.assign. 
For example, W=-1 and b=1 are the optimal parameters for our model. We can change W and b accordingly:
"""

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


"""
We guessed the "perfect" values of W and b, but the whole point of machine learning is to find the correct model 
parameters automatically. We will show how to accomplish this in the next section.
"""

"""
tf.train API
A complete discussion of machine learning is out of the scope of this tutorial. 
However, TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function. 
The simplest optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative 
of loss with respect to that variable. In general, computing symbolic derivatives manually is tedious and error-prone. 
Consequently, TensorFlow can automatically produce derivatives given only a description of the model using 
the function tf.gradients. For simplicity, optimizers typically do this for you. For example,

"""

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

"""
Now we have done actual machine learning! Although this simple linear regression model does not require much TensorFlow
 core code, more complicated models and methods to feed data into your models necessitate more code. 
 Thus, TensorFlow provides higher level abstractions for common patterns, structures, and functionality. 
 We will learn how to use some of these abstractions in the next section.
"""







