#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  line_regression.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-09-27 16:31:06
#  Last Modified:  2019-09-29 00:05:59
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

xdata = xdata.reshape(CNT, 1)
ydata = ydata.reshape(CNT, 1)

w = tf.Variable([0.1], dtype=tf.float32, name='w')
b = tf.Variable([-0.1], dtype=tf.float32, name='b')

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

lmodel = w * x + b

loss = tf.reduce_sum(tf.square(lmodel - y), name='loss')
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(train, {x: xdata, y: ydata})
    print('W=', sess.run(w), 'b=', sess.run(b), 'loss=', sess.run(loss))
