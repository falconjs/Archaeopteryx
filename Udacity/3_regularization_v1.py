# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from datetime import datetime as dt
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import math

# Config the matplotlib backend as plotting inline in IPython
# %matplotlib inline

# Define the store data to elsewhere
data_root = '../../input/notMNIST/'

pickle_file = data_root + 'notMNIST.pickle'

label_dic = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J'}

print('pickle file location: %s' % pickle_file)

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset_rw = save['train_dataset']
    train_labels_rw = save['train_labels']
    valid_dataset_rw = save['valid_dataset']
    valid_labels_rw = save['valid_labels']
    test_dataset_rw = save['test_dataset']
    test_labels_rw = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset_rw.shape, train_labels_rw.shape)
    print('Validation set', valid_dataset_rw.shape, valid_labels_rw.shape)
    print('Test set', test_dataset_rw.shape, test_labels_rw.shape)

# display the dataset and labels
img_idx_list = list(range(0,20))
img_idx_len = len(img_idx_list)

col_num = 10
row_num = math.ceil(img_idx_len / col_num)

print("%d, %d" % (row_num, col_num))

plt.figure(figsize=(18, 2 * row_num))
for i, idx in enumerate(img_idx_list):
    print(label_dic[train_labels_rw[idx]], end=' ')
    if i % col_num + 1 == col_num:
        print()
    plt.subplot(row_num, col_num, i+1)
    plt.imshow(train_dataset_rw[idx], cmap='gray')

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)

    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset_rw, train_labels_rw)
valid_dataset, valid_labels = reformat(valid_dataset_rw, valid_labels_rw)
test_dataset, test_labels = reformat(test_dataset_rw, test_labels_rw)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


# Define a fully connected layer
def fc_layer(input_x, channels_in, channels_out, layer_name='Full_Connection_Layer'):
    with tf.name_scope(layer_name):
        # It is not a good idea to set initial value as zero
        # It will cause problem during the learning activity
        # w = tf.Variable(tf.zeros([channels_in, channels_out]))

        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random values following a (truncated)
        # normal distribution.

        with tf.variable_scope(layer_name):
            # weights = tf.Variable(tf.truncated_normal([channels_in, channels_out], seed=1), name='W')
            weights = tf.get_variable(name='W', shape=[channels_in, channels_out], \
                                      initializer=tf.truncated_normal_initializer(seed=1))
            # The biases get initialized to zero.
            # biases = tf.Variable(tf.zeros([channels_out]), name='B')
            biases = tf.get_variable(name='B', shape=[channels_out], \
                                     initializer=tf.zeros_initializer())

        fc_conn = tf.matmul(input_x, weights)
        act = tf.nn.relu(fc_conn + biases)

        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        # tf.summary.histogram("fc_conn", fc_conn)

        return act


# Define a Convolutional layer
def conv_layer(input_x, channels_in, channels_out, layer_name='Convolutional_Layer'):
    with tf.name_scope(layer_name):
        with tf.variable_scope(layer_name):
            weights = tf.Variable(tf.zeros([5, 5, channels_in, channels_out]), name='W')
            biases = tf.Variable(tf.zeros(channels_out), name='B')
        conv_conn = tf.nn.conv2d(input_x, weights, strides=[1, 1, 1, 1], padding='SAME')
        act = tf.nn.relu(conv_conn + biases)

        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        # tf.summary.histogram("conv_conn", conv_conn)

        return act


# build the network graph

graph_1 = tf.Graph()
with graph_1.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    with tf.name_scope('Input_X'):
        tf_train_dataset = tf.placeholder(tf.float32, shape=(None, image_size * image_size), name='Train_X')
        # tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size), name='Train_X')
        # tf_valid_dataset = tf.constant(valid_dataset, name='Valid_X')
        # tf_test_dataset = tf.constant(test_dataset, name='Test_X')

    with tf.name_scope('Labels_y'):
        # tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='T_y')
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels), name='T_y')

    with tf.name_scope('Beta_L2_regu'):
        tf_beta_l2_regu = tf.placeholder(tf.float32, name='Beta_L2_regu')

    with tf.name_scope('Data_reshape_for_image'):
        # image for display purpose
        tf_train_ds_image = tf.reshape(tf_train_dataset, [-1, 28, 28, 1])
        # tensorboard logging
        tf.summary.image('input', tf_train_ds_image, 3)

    # Create the network
    full_conn_1 = fc_layer(tf_train_dataset, image_size * image_size, num_labels, layer_name='fc_conn_1')

    logits = full_conn_1

    with tf.variable_scope('fc_conn_1', reuse=True):
        fc_conn_w = tf.get_variable("W", [image_size * image_size, num_labels])

    with tf.name_scope('loss_function'):
        l2_loss = tf.nn.l2_loss(fc_conn_w, name='l2_loss')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels,
                                                                         logits=logits), name='cross_entropy') \
               + tf_beta_l2_regu * l2_loss
        tf.summary.scalar('beta_l2_regu', tf_beta_l2_regu)
        tf.summary.scalar('cross_entropy', loss)
        tf.summary.scalar('l2_loss', l2_loss)

    # Optimizer.
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Accuracy
    with tf.name_scope('Accuracy'):
        prediction = logits
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(tf_train_labels, 1))
        accuracy_res = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy_res', accuracy_res)

# Run the model

batch_size = 128
num_steps = 3001
beta_l2_regu = 0.01

with tf.Session(graph=graph_1) as session:
    # Initialize all the variables
    # session.run(tf.global_variables_initializer())
    tf.global_variables_initializer().run()
    print("Initialized")

    # Make the tensorboard log writer
    session_log_dir = "logs/3_1/" + dt.today().strftime('%Y%m%d%H%M%S')
    writer = tf.summary.FileWriter(session_log_dir)
    print("Logging Directory : %s" % session_log_dir)

    writer.add_graph(session.graph)

    # Merge all the tf summary
    merged_summary = tf.summary.merge_all()

    # Data Set
    # Minibatch will be built in loop
    valid_feed_dict = {tf_train_dataset: valid_dataset, tf_train_labels: valid_labels, tf_beta_l2_regu: beta_l2_regu}
    test_feed_dict = {tf_train_dataset: test_dataset, tf_train_labels: test_labels, tf_beta_l2_regu: beta_l2_regu}

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
        train_feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, tf_beta_l2_regu: beta_l2_regu}

        if step % 5 == 0:
            s = session.run(merged_summary, feed_dict=train_feed_dict)
            writer.add_summary(s, step)

        _, l, train_prediction = session.run([optimizer, loss, prediction], feed_dict=train_feed_dict)

        if step % 500 == 0:
            # Predictions for the validation, and test data.

            valid_prediction = session.run(logits, feed_dict=valid_feed_dict)
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(train_prediction, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction, valid_labels))

    test_prediction = session.run(logits, feed_dict=test_feed_dict)
    print("Test accuracy: %.1f%%" % accuracy(test_prediction, test_labels))
    writer.close()


"""
Problem 2
Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?
"""

# build the network graph

graph_2 = tf.Graph()
with graph_2.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    with tf.name_scope('Input_X'):
        tf_train_dataset = tf.placeholder(tf.float32, shape=(None, image_size * image_size), name='Train_X')
        # tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size), name='Train_X')
        # tf_valid_dataset = tf.constant(valid_dataset, name='Valid_X')
        # tf_test_dataset = tf.constant(test_dataset, name='Test_X')

    with tf.name_scope('Labels_y'):
        # tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='T_y')
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels), name='T_y')

    with tf.name_scope('Beta_L2_regu'):
        tf_beta_l2_regu = tf.placeholder(tf.float32, name='Beta_L2_regu')

    with tf.name_scope('Data_reshape_for_image'):
        # image for display purpose
        tf_train_ds_image = tf.reshape(tf_train_dataset, [-1, 28, 28, 1])
        # tensorboard logging
        tf.summary.image('input', tf_train_ds_image, 3)

    # Create the network
    full_conn_1 = fc_layer(tf_train_dataset, image_size * image_size, num_labels, layer_name='fc_conn_1')

    logits = full_conn_1

    with tf.variable_scope('fc_conn_1', reuse=True):
        fc_conn_w = tf.get_variable("W", [image_size * image_size, num_labels])

    with tf.name_scope('loss_function'):
        l2_loss = tf.nn.l2_loss(fc_conn_w, name='l2_loss')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels,
                                                                         logits=logits), name='cross_entropy') \
               + tf_beta_l2_regu * l2_loss
        tf.summary.scalar('beta_l2_regu', tf_beta_l2_regu)
        tf.summary.scalar('cross_entropy', loss)
        tf.summary.scalar('l2_loss', l2_loss)

    # Optimizer.
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Accuracy
    with tf.name_scope('Accuracy'):
        prediction = logits
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(tf_train_labels, 1))
        accuracy_res = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy_res', accuracy_res)

# Run the model

batch_size = 1
num_steps = 3001

with tf.Session(graph=graph_2) as session:
    # Initialize all the variables
    # session.run(tf.global_variables_initializer())
    tf.global_variables_initializer().run()
    print("Initialized")

    # Make the tensorboard log writer
    session_log_dir = "logs/3_2/" + dt.today().strftime('%Y%m%d%H%M%S')
    writer = tf.summary.FileWriter(session_log_dir)
    print("Logging Directory : %s" % session_log_dir)

    writer.add_graph(session.graph)

    # Merge all the tf summary
    merged_summary = tf.summary.merge_all()

    # Data Set
    # Minibatch will be built in loop
    valid_feed_dict = {tf_train_dataset: valid_dataset, tf_train_labels: valid_labels, tf_beta_l2_regu: beta_l2_regu}
    test_feed_dict = {tf_train_dataset: test_dataset, tf_train_labels: test_labels, tf_beta_l2_regu: beta_l2_regu}

    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    # offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    offset = 145

    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    print(batch_data.shape, batch_labels.shape)

    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    train_feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, tf_beta_l2_regu: beta_l2_regu}

    for step in range(num_steps):
        if step % 5 == 0:
            s = session.run(merged_summary, feed_dict=train_feed_dict)
            writer.add_summary(s, step)

        _, l, train_prediction = session.run([optimizer, loss, prediction], feed_dict=train_feed_dict)

        if step % 500 == 0:
            # Predictions for the validation, and test data.
            valid_prediction = session.run(logits, feed_dict=valid_feed_dict)
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(train_prediction, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction, valid_labels))

    test_prediction = session.run(logits, feed_dict=test_feed_dict)
    print("Test accuracy: %.1f%%" % accuracy(test_prediction, test_labels))
    writer.close()


