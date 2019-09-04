#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  001.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-03-05 15:50:29
#  Last Modified:  2019-09-02 16:32:42
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

import tensorflow as tf
sess = tf.InteractiveSession()

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 2.0], name='b')

res = a + b
print(res.eval())
