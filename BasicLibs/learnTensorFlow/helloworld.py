# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/8
    Desc : 
'''

import tensorflow as tf

import numpy as np

w1 = tf.Variable(tf.random_normal([5]))[0]

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(w1))
