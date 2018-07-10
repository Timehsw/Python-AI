# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/10
    Desc : 编写一段代码，实现动态的更新变量的维度数目
'''

import tensorflow as tf

# 1. 定义一个不定形状的变量

x = tf.Variable(
    initial_value=[],  # 给定一个空值
    dtype=tf.float32,
    trainable=False,
    validate_shape=False  # 设置为true,表示在变量更新的时候,进行shape的检查,默认为True
)

# 2. 变量更改
concat = tf.concat([x, [0.0, 0.0]], axis=0)
assign_op = tf.assign(x, concat, validate_shape=False)

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
