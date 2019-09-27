#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  m.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-09-02 16:34:30
#  Last Modified:  2019-09-02 23:58:22
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

x = np.arange(1, 11)
y = 2 * x + 5
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x, y)
plt.show()

print(matplotlib.get_backend())
