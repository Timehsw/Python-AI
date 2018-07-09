# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/8
    Desc : 
'''

import tensorflow as tf

import numpy as np

a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

# c=tf.add(a,b)
c = tf.multiply(a, b)
with tf.Session() as session:
    res = session.run(c, feed_dict={a: 2, b: 5})
    print(res)
