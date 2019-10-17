#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  loss_test.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-09-28 14:43:49
#  Last Modified:  2019-09-28 14:51:18
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

import numpy as np
import math


def loss_test(yr, yp):
    loss = -1 * (yr * math.log(yp)) + (1 - yr) * math.log(1 - yp)
    return loss


print(loss_test(0.5, 0.19))
print(loss_test(0.5, 0.501))
