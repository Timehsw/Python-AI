# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/8
    Desc : 
'''

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
