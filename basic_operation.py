#!/usr/bin/env python
#-*- coding:utf-8 -*-
#       FileName:  basic_operation.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2018-06-29 11:03:55
#  Last Modified:  2019-03-05 15:41:02
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

from __future__ import print_function
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print("a=2,b=3")
    print("addition with constant: %i" % sess.run(a + b))
    print("Multiplication with constants: %i" % sess.run(a * b))

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print("add with :%i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("mul %s" % sess.run(mul, feed_dict={a: 2, b: 3}))

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
