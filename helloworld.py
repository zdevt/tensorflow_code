#!/usr/bin/env python
#-*- coding:utf-8 -*-
#       FileName:  helloworld.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2018-06-29 10:52:34
#  Last Modified:  2018-06-29 11:03:34
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

from __future__ import print_function

import tensorflow as tf

hello = tf.constant("Hello ")

sess = tf.Session()

print(sess.run(hello))
