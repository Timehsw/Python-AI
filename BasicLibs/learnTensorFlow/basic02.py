# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/8
    Desc : 使用默认图进行编程
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
