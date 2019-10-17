#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  cpugpu_check.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-08-10 22:21:15
#  Last Modified:  2019-10-07 22:19:53
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

import os
from tensorflow.python.client import device_lib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"

if __name__ == "__main__":
    print(device_lib.list_local_devices())
