# ==============================================================================
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    # print(FLAGS.data_dir)
    # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    """
    Each image is 28 pixels by 28 pixels. We can interpret this as a big array of numbers:
    We can flatten this array into a vector of 28x28 = 784 numbers. It doesn't matter how we flatten the array, 
    as long as we're consistent between images. From this perspective, the MNIST images are just a bunch of points 
    in a 784-dimensional vector space, with a very rich structure (warning: computationally intensive visualizations).
    
    The result is that mnist.train.images is a tensor (an n-dimensional array) with a shape of [55000, 784]. 
    The first dimension is an index into the list of images and the second dimension is the index for each pixel 
    in each image. Each entry in the tensor is a pixel intensity between 0 and 1, for a particular pixel 
    in a particular image.
    
    Each image in MNIST has a corresponding label, a number between 0 and 9 representing the digit drawn in the image.

    For the purposes of this tutorial, we're going to want our labels as "one-hot vectors". 
    A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the
    Nth digit will be represented as a vector which is 1 in the Nth dimension. For example, 3 would be . 
    Consequently, mnist.train.labels is a [55000, 10] array of floats.
    """

    # Create the model Implementing the Regression
    x = tf.placeholder(tf.float32, [None, 784])

    """
    x isn't a specific value. It's a placeholder, a value that we'll input when we ask TensorFlow to run a computation. 
    We want to be able to input any number of MNIST images, each flattened into a 784-dimensional vector. 
    We represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784]. 
    (Here None means that a dimension can be of any length.)
    """

    """
    We also need the weights and biases for our model. We could imagine treating these like additional inputs, 
    but TensorFlow has an even better way to handle it: Variable. A Variable is a modifiable tensor that lives in 
    TensorFlow's graph of interacting operations. It can be used and even modified by the computation. 
    For machine learning applications, one generally has the model parameters be Variables.
    """

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    """
    We create these Variables by giving tf.Variable the initial value of the Variable: in this case, 
    we initialize both W and b as tensors full of zeros. Since we are going to learn W and b, it doesn't matter 
    very much what they initially are.

    Notice that W has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors by it to 
    produce 10-dimensional vectors of evidence for the difference classes. 
    b has a shape of [10] so we can add it to the output.

    We can now implement our model. It only takes one line to define it!
    """

    y = tf.matmul(x, W) + b

    """
    First, we multiply x by W with the expression tf.matmul(x, W). This is flipped from when we multiplied them 
    in our equation, where we had Wx, as a small trick to deal with x being a 2D tensor with multiple inputs. 
    We then add b, and finally apply tf.nn.softmax.
    """

    """
    That's it. It only took us one line to define our model, after a couple short lines of setup. 
    That isn't because TensorFlow is designed to make a softmax regression particularly easy: it's just a very flexible 
    way to describe many kinds of numerical computations, from machine learning models to physics simulations. 
    And once defined, our model can be run on different devices: your computer's CPU, GPUs, and even phones!
    """

    # ============== Training =================

    """
    In order to train our model, we need to define what it means for the model to be good. Well, actually, 
    in machine learning we typically define what it means for a model to be bad. We call this the cost, or the loss, 
    and it represents how far off our model is from our desired outcome. We try to minimize that error, 
    and the smaller the error margin, the better our model is.

    One very common, very nice function to determine the loss of a model is called "cross-entropy." 
    Cross-entropy arises from thinking about information compressing codes in information theory 
    but it winds up being an important idea in lots of areas, from gambling to machine learning. 
    It's defined as:
    
    H (y) on y'  = - sum(i) of ( y'(i) * log( y(i) ) ) 

    Where y is our predicted probability distribution, and y' is the true distribution 
    (the one-hot vector with the digit labels). In some rough sense, the cross-entropy is measuring 
    how inefficient our predictions are for describing the truth. Going into more detail about cross-entropy 
    is beyond the scope of this tutorial, but it's well worth understanding.

    """

    """
    To implement cross-entropy we need to first add a new placeholder to input the correct answers:
    """
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    """
    First, tf.log computes the logarithm of each element of y. 
    Next, we multiply each element of y_ with the corresponding element of tf.log(y). 
    Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter. 
    Finally, tf.reduce_mean computes the mean over all the examples in the batch.
    """


    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.

    """
    Note that in the source code, we don't use this formulation, because it is numerically unstable. 
    Instead, we apply tf.nn.softmax_cross_entropy_with_logits on the unnormalized logits 
    (e.g., we call softmax_cross_entropy_with_logits on tf.matmul(x, W) + b), 
    because this more numerically stable function internally computes the softmax activation. 
    In your code, consider using tf.nn.softmax_cross_entropy_with_logits instead.
    """

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    """
    Now that we know what we want our model to do, it's very easy to have TensorFlow train it to do so. 
    Because TensorFlow knows the entire graph of your computations, it can automatically use the backpropagation 
    algorithm to efficiently determine how your variables affect the loss you ask it to minimize. 
    Then it can apply your choice of optimization algorithm to modify the variables and reduce the loss.
    """

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    """
    In this case, we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm 
    with a learning rate of 0.5. Gradient descent is a simple procedure, where TensorFlow simply shifts 
    each variable a little bit in the direction that reduces the cost. But TensorFlow also provides 
    many other optimization algorithms: using one is as simple as tweaking one line.
    """

    """
    We can now launch the model in an InteractiveSession:
    """

    sess = tf.InteractiveSession()

    """
    We first have to create an operation to initialize the variables we created:
    """

    tf.global_variables_initializer().run()

    """Let's train -- we'll run the training step 1000 times!"""
    # Train
    for _ in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    """
    Each step of the loop, we get a "batch" of one hundred random data points from our training set. 
    We run train_step feeding in the batches data to replace the placeholders.

    Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent.
    Ideally, we'd like to use all our data for every step of training because that would give us a better sense of 
    what we should be doing, but that's expensive. So, instead, we use a different subset every time. 
    Doing this is cheap and has much of the same benefit.
    """

    # Evaluating Our Model

    # Test trained model
    """
    Well, first let's figure out where we predicted the correct label. tf.argmax is an extremely useful function 
    which gives you the index of the highest entry in a tensor along some axis. 
    For example, tf.argmax(y,1) is the label our model thinks is most likely for each input, 
    while tf.argmax(y_,1) is the correct label. We can use tf.equal to check if our prediction matches the truth.
    """

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    """
    That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and
    then take the mean. For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.
    """

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    """Finally, we ask for our accuracy on our test data."""

    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))

    """
    This should be about 92%.

    Is that good? Well, not really. In fact, it's pretty bad. This is because we're using a very simple model. 
    With some small changes, we can get to 97%. The best models can get to over 99.7% accuracy! 
    (For more information, have a look at this list of results.)
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='MNIST_data/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)