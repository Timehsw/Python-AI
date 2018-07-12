# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/10
    Desc : 实现一个累加器，并且每一步均输出累加器的结果值。
'''

import tensorflow as tf

# 1. 定义一个变量
x = tf.Variable(0, dtype=tf.int32, name='v_x')

# 2. 变量的更新
assign_op = tf.assign(ref=x, value=x + 1)

# 3. 变量的初始化操作

x_init_op = tf.global_variables_initializer()

# 4.运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as session:
    # 变量初始化
    session.run(x_init_op)

    # 模拟迭代更新累加器
    for i in range(5):
        r_x = session.run(x)
        print(r_x)
        session.run(assign_op)
