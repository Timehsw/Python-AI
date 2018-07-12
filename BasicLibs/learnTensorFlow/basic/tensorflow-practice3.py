# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/10
    Desc : 实现一个求解阶乘的代码
'''

import tensorflow as tf

# 1. 定义一个变量
sum = tf.Variable(1, dtype=tf.int32)

# 2. 定义一个占位符
i = tf.placeholder(dtype=tf.int32)

# 3. 变量的更新
assign_op = tf.assign(sum, sum * i)

# 4. 变量的初始化操作
x_init_op = tf.global_variables_initializer()

# 5.运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as session:
    # 变量初始化
    session.run(x_init_op)

    # 模拟迭代更新累加器
    for j in range(1, 6):
        session.run(assign_op, feed_dict={i: j})
    print('5!={}'.format(session.run(sum)))
