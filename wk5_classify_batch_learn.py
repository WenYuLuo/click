import tensorflow as tf
import find_click
import numpy as np
import time
import matplotlib.pyplot as plt


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


n_batch = 2
n_class = 3
# n_feature = 4


x = tf.placeholder("float", [None, n_class])
# y = tf.placeholder("float", [None, n_class])

temp_y = tf.reshape(x, [-1, n_batch, n_class])
# multi_y = tf.reshape(temp_y, [n_batch, -1])

temp_y_t = tf.transpose(temp_y, perm=[0, 2, 1])

multi_y = tf.reshape(temp_y_t, [-1, n_batch])
multi_y_t = tf.transpose(multi_y, perm=[1, 0])

W_fuse = weight_variable([1, n_batch])
b_fuse = bias_variable([n_class])
#
#
# c = tf.matmul(W_fuse, temp_y)

c = tf.reshape(tf.matmul(W_fuse, multi_y_t), [-1, n_class]) + b_fuse

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    a = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
    print('a:', a)
    temp_y_out, multi_y_out, b_fuse, c_out = sess.run((temp_y_t, multi_y_t, b_fuse, c), feed_dict={x: a})

    # temp_y_out = sess.run(temp_y, feed_dict={x: a})
    print(temp_y_out)
    print(multi_y_out)
    print(b_fuse)
    print(c_out)
