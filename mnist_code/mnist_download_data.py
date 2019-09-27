#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  mnist_data.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-09-04 14:30:34
#  Last Modified:  2019-09-04 14:30:41
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
