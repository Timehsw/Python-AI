# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/8
    Desc : 不使用默认图进行编程
'''

import tensorflow as tf

# 1. 定义常量矩阵a和矩阵b
a = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
b = tf.constant([5, 6, 7, 8], dtype=tf.int32, shape=[2, 2])

print(type(a))
print(type(b))

# 2. 以a和b作为输入,进行矩阵乘法(matmul)操作
c = tf.matmul(a, b)

print(type(c))

print('变量a是否在默认图中:{}'.format(a.graph is tf.get_default_graph()))

graph = tf.Graph()
with graph.as_default():
    # 此时在这个代码块中,使用的就是新定义的图graph
    # 相当于把默认图换成了graph
    # 但是只在这个代码块中有效
    d = tf.constant(5.0)
    print('变量d是否在新图graph中:{}'.format(d.graph is graph))

    print(d.graph is tf.get_default_graph())

    pass


# 但是只在这个代码块中有效,此时输出为false

print(d.graph is tf.get_default_graph())


with tf.Graph().as_default() as g2:
    e = tf.constant(5.0)
    print('变量e是否在新图g2中:{}'.format(e.graph is g2))

# 这段代码是错误的用法,记住,不能使用两个图中的变量进行操作

