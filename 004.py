#!/usr/bin/env python
#-*- coding:utf-8 -*-
#       FileName:  003.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-05-25 14:02:08
#  Last Modified:  2019-06-05 10:54:20
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

import tensorflow as tf
import numpy as np

a = tf.zeros([3, 4])
b = tf.ones([2, 3])
c = tf.ones_like(b)

d = tf.constant(-3.14, shape=[3, 5])

x = tf.linspace(1.0, 8.0, 5)

x = tf.random_normal([3, 3], 0.0, 1.0, tf.float32, 1)

z = tf.Variable(x)
y = tf.random_shuffle(z)

state = tf.Variable(0)
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)

saver = tf.train.Saver()

an = np.zeros((3, 3))
tan = tf.convert_to_tensor(an)

o1 = tf.placeholder(tf.float32)
o2 = tf.placeholder(tf.float32)
o = tf.multiply(o1,o2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    '''
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
    save_path = saver.save(sess, "./test")
    print(save_path)
    print(tan.eval())
    '''
    print(sess.run([o], {o1: [3.], o2: [5.]}))
