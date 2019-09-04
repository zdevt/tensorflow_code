#!/usr/bin/env python
#-*- coding:utf-8 -*-
#       FileName:  tf_1_2.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-03-06 14:30:01
#  Last Modified:  2019-03-06 14:33:29
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

import tensorflow as tf

var = tf.Variable(0,name="myvar")

con_var = tf.constant(1)

new_var = tf.add(var,con_var)

init = tf.global_variables_initializer()

sess=tf.Session()

with tf.Session() as sess:
    sess.run(init)
    print("var: ",sess.run(var))
    print("con_var: ",sess.run(con_var))
    print("new_var: ",sess.run(new_var))




