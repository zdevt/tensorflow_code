#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  ts_estimator.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-09-27 22:16:22
#  Last Modified:  2019-09-27 22:16:33
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:
import numpy as np
import tensorflow as tf

# 定义特性列，线性模型中特性是列是x，shape=[1]，因此定义如下：
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# 使用tf.estimator内置的LinearRegressor来完成线性回归算法
# tf.estimator提供了很多常规的算法模型以便用户调用，不需要用户自己重复造轮子
# 到底为止，短短两行代码我们的建模工作就已经完成了
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# 有了模型之后，我们要使用模型完成训练->评估->预测这几个步骤
# 训练数据依旧是(1.,0.)，(2.,-1.)，(3.,-2.)，(4.,-3.)这几个点，拆成x和y两个维度的数组
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])

# 评估数据为(2.,-1.01)，(5.,-4.1)，(8.,-7.)，(1.,0.)这四个点，同样拆分成x和y两个维度的数组
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7., 0.])

# 用tf.estimator.numpy_input_fn方法生成随机打乱的数据组，每组包含4个数据
input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
# 循环1000次训练模型
estimator.train(input_fn=input_fn, steps=1000)

# 生成训练数据，分成1000组，每组4个数据
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
# 生成评估数据，分成1000组，每组4个数据
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# 训练数据在模型上的预测准确率
train_metrics = estimator.evaluate(input_fn=train_input_fn)
# 评估数据在模型上的预测准确率
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r" % train_metrics)
print("eval metrics: %r" % eval_metrics)
