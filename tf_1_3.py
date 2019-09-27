#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  tf_1_3.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-03-06 16:47:09
#  Last Modified:  2019-09-04 14:25:31
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
