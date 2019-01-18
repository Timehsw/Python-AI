# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-01-17
    Desc : 
    Note : 
'''

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([5]))[0]
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(w1))
