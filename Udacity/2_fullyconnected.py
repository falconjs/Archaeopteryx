#
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

"""
First reload the data we generated in 1_notmnist.ipynb.
"""

data_root = '../../input/notMNIST/' # Change me to store data elsewhere
pickle_file = data_root + 'notMNIST.pickle'

print('pickle file location: %s' % pickle_file)

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


"""
Reformat into a shape that's more adapted to the models we're going to train:

data as a flat matrix,
labels as float 1-hot encodings.
"""

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)

    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

"""
We're first going to train a multinomial logistic regression using simple gradient descent.

TensorFlow works like this:

First you describe the computation that you want to see performed: what the inputs, the variables, 
and the operations look like. These get created as nodes over a computation graph. 
This description is all contained within the block below:

with graph.as_default():
    ...
Then you can run the operations on this graph as many times as you want by calling session.run(), 
providing it outputs to fetch from the graph that get returned. This runtime operation is all 
contained in the block below:

with tf.Session(graph=graph) as session:
    ...
Let's load all the data into TensorFlow and build the computation graph corresponding to our training:
"""

"""
Turn the logistic regression example with SGD into a 1-hidden layer neural network with rectified 
linear units nn.relu() and 1024 hidden nodes. This model should improve your validation / test accuracy.
"""

batch_size = 128


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    with tf.name_scope('Input_data_X'):
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size), name='Train_X')
        tf_valid_dataset = tf.constant(valid_dataset, name='Valid_X')
        tf_test_dataset = tf.constant(test_dataset, name='Test_X')

        # image for display purpose
        tf_train_ds_image = tf.reshape(tf_train_dataset, [-1, 28, 28, 1])
        tf.summary.image('input', tf_train_ds_image, 3)

    with tf.name_scope('Labels_y'):
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='T_y')

    # Variables.
    with tf.name_scope('variables'):
        weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels], name='Tnc_Norm'), name='weights')
        biases = tf.Variable(tf.zeros([num_labels]), name='biases')

    # Training computation.
    with tf.name_scope('FC_L1'):
        logits = tf.matmul(tf_train_dataset, weights) + biases
        act_logits = tf.nn.relu(logits)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("act_logits", act_logits)

    with tf.name_scope('loss_function'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels,
                                                                         logits=act_logits), name='cross_entropy')

    # Optimizer.
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    with tf.name_scope('Prediction'):
        train_prediction = tf.nn.softmax(tf.matmul(tf_train_dataset, weights) + biases)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    # Accuracy
    with tf.name_scope('Accuracy_res'):
        correct_prediction = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(tf_train_labels, 1))
        accuracy_res = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('cross_entropy', loss)
    tf.summary.scalar('accuracy_res', accuracy_res)


num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/3")
    writer.add_graph(session.graph)

    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

        if step % 5 == 0:
            s = session.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(s, step)

        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

