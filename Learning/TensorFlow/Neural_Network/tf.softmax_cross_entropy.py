"""
https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits_v2

Defined in tensorflow/python/ops/nn_ops.py.

See the guide: Neural_Network > Classification

Computes softmax cross entropy between logits and labels.

Measures the probability error in discrete classification tasks in which
 the classes are mutually exclusive (each entry is in exactly one class).
 For example, each CIFAR-10 image is labeled with one and only one label:
 an image can be a dog or a truck, but not both.

"""

import tensorflow as tf

tf_train_labels = tf.constant(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]],
     dtype=tf.float32
)

logits_the_correct = tf.constant(
    [[2, 0, 0],
     [0, 2, 0],
     [0, 0, 2],
     [5, 0, 0],
     [0, 6, 0],
     [0, 0, 7]],
     dtype=tf.float32
)

logits_on_side = tf.constant(
    [[0.2, -0.2, 0],
     [0.2, -0.2, 0],
     [0.6, -0.6, 0],
     [0.4, -0.4, 0],
     [0.3, -0.3, 0],
     [0.2, -0.2, 0]],
     dtype=tf.float32
)

logits_no_decision = tf.constant(
    [[2, -2, 0],
     [2, -2, 0],
     [6, -6, 0],
     [4, -4, 0],
     [3, -3, 0],
     [2, -2, 0]],
     dtype=tf.float32
)

sce_1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits_the_correct,
                                                                   name='cross_entropy')
sce_2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits_on_side,
                                                                   name='cross_entropy')
sce_3 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits_no_decision,
                                                                   name='cross_entropy')
sum_2 = tf.reduce_sum(sce_2)

sess = tf.Session()
print(sess.run(sce_1))
print(sess.run(sce_2))
print(sess.run(sce_3))
# [0.23954484 0.23954484 0.23954473 0.0133859  0.00494519 0.00182212]
# [0.9119015 1.3119015 1.2151889 0.7512505 1.4283903 1.1119015]
# [0.1429317  4.142932   6.002482   0.01847933 6.0509458  2.1429317 ]

print(sess.run(sum_2))
print(sess.run(tf.reduce_sum(sce_3)))
# 6.7305346
# 18.500702
