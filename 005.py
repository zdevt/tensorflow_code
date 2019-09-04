#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  003.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-05-25 14:02:08
#  Last Modified:  2019-09-02 16:33:13
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
num_points = 1000
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

plt.scatter(x_data, y_data, c='r')
plt.show()

W = tf.Variable(tf.random_uniform([1], -1, 1.0), name='W')
b = tf.Variable(tf.zeros([1]), name='b')

y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss, name='train')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(50):
        sess.run(train)
        print('W=', sess.run(W), 'b=', sess.run(b), 'loss=', sess.run(loss))
