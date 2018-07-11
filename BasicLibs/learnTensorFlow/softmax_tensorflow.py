# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/11
    Desc : tensorflow 之 softmax
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 构造数据
np.random.seed(28)
N = 100
x = np.linspace(0, 6, N) + np.random.normal(loc=0.0, scale=2.0, size=N)
# 让y波动一下,整体不变
y = 14 * x - 7 + np.random.normal(loc=0.0, scale=1.0, size=N)

# 将x和y设置为矩阵
x.shape = -1, 1
y.shape = -1, 1

# 2.模型构建 y=wx+b
# 定义一个变量w和b
# random_uniform:产生服从均匀分布的随机数列
w = tf.Variable(initial_value=tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0, ), name='w')
b = tf.Variable(initial_value=tf.zeros([1]), name='b')

# 构建一个预测值
y_hat = w * x + b

# 构建一个损失函数
# 以mse作为损失函数,预测值与实际值的平方和
loss = tf.reduce_mean(tf.square(y_hat - y), name='loss')

# 以随机梯度下降的方式来优化损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
# 在优化的过程中,是让损失函数最小化
train = optimizer.minimize(loss, name='train')

# 全局变量更新
init_op = tf.global_variables_initializer()


def print_into(r_w, r_b, r_loss):
    print('w={},b={},loss={}'.format(r_w, r_b, r_loss))


# 运行
with tf.Session() as session:
    # 初始化
    session.run(init_op)

    # 输出初始化的w,b,loss
    r_w, r_b, r_loss = session.run([w, b, loss])
    print_into(r_w, r_b, r_loss)

    # 进行训练(100次)
    for step in range(100):
        # 模型训练
        session.run(train)
        # 输出训练后的w,b,loss
        r_w, r_b, r_loss = session.run([w, b, loss])
        print_into(r_w, r_b, r_loss)
