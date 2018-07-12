# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/11
    Desc : tensorflow基础代码---常量变量
'''

import tensorflow as tf

# 1. 定义常量矩阵a和矩阵b
a = tf.Variable(initial_value=5, dtype=tf.int32, name='a')
# a = tf.constant(value=10, dtype=tf.int32)
b = tf.constant(value=[5, 6, 7, 8], dtype=tf.int32, shape=[2, 2], name='b')

print(type(a))
print(type(b))

print(a)
print(b)

g = tf.placeholder(dtype=tf.int32, shape=[1], name='g1')

f = tf.assign(a, a * 2)


init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    session.run(f)
    print(session.run(a))
    # print('res:{}'.format(res))
