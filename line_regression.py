#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  line_regression.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-09-27 16:31:06
#  Last Modified:  2019-09-27 21:57:27
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

import tensorflow as tf
import numpy as np

CNT = 5000
xdata = np.random.rand(CNT)
ydata = 0.3 * xdata - 0.2
# print(xdata)
# print(ydata)

w = tf.Variable([0.1], dtype=tf.float32)
b = tf.Variable([-0.1], dtype=tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

lmodel = w * x + b

loss = tf.reduce_sum(tf.square(lmodel - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(CNT):
        sess.run(train, {x: xdata[i], y: ydata[i]})

    currW, currB, currLoss = sess.run([w, b, loss], {x: xdata, y: ydata})
    print(currW, currB, currLoss)
