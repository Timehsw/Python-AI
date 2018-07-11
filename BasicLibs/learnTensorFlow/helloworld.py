# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/8
    Desc : 
'''

import tensorflow as tf

import numpy as np

aa = tf.random_uniform((2,2), 0, 0.5)
print(aa)
# a = tf.Variable(initial_value=tf.random_uniform([1],-1.0,1.0),name='w')
# c=tf.add(a,b)
# c = tf.multiply(a, b)
with tf.Session() as session:
    res = session.run(aa)
    print(res)
