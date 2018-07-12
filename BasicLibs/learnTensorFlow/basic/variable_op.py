# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/9
    Desc : 
'''

import tensorflow as tf

# 1. 定义一个变量,必须给定初始值
a = tf.Variable(initial_value=3.0, dtype=tf.float32)

# 2. 定义一个张量
b = tf.constant(value=2.0, dtype=tf.float32)

c = tf.add(a, b)

# 3. 进行初始化操作(推荐:使用全局所有变量初始化api)
# 相当于在图中加入一个初始化全局变量的操作
init_op = tf.global_variables_initializer()

# tf.initialize_all_variables()
# a.initial_value()


# 3. 图的运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    # 运行init_op进行变量初始化,一定要放到所有运行操作之前
    session.run(init_op)
    # 获取操作的结果.两种方式
    print("result:{}".format(session.run(c)))
    print("result:{}".format(c.eval()))
