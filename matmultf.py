#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  003.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-05-25 14:02:08
#  Last Modified:  2019-09-27 21:58:32
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

import tensorflow as tf

w = tf.Variable([[0.5, 1.0]])
x = tf.Variable([[2.0], [1.0]])

y = tf.matmul(w, x)
print(w)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(y.eval())
