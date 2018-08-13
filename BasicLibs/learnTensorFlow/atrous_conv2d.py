# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/8/13
    Desc : atrous_conv2d
    Note : 
'''

import numpy as np
import tensorflow as tf

x_image = tf.placeholder(tf.float32, shape=[5, 5])
x = tf.reshape(x_image, [1, 5, 5, 1])

# Filter: W
W_cpu = np.array([[3, 1, 1], [0, -2, 0], [0, -1, 1]], dtype=np.float32)
W = tf.Variable(W_cpu)
W = tf.reshape(W, [3, 3, 1, 1])

strides = [1, 1, 1, 1]  # 没用到
padding = 'VALID'

y = tf.nn.atrous_conv2d(x, W, 2, padding)

x_data = np.array([[1, 0, 0, 0, 0], [2, 1, 1, 2, 1], [1, 1, 2, 2, 0], [2, 2, 1, 0, 0], [2, 1, 2, 1, 1]],
                  dtype=np.float32)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    x = (sess.run(x, feed_dict={x_image: x_data}))
    W = (sess.run(W, feed_dict={x_image: x_data}))
    y = (sess.run(y, feed_dict={x_image: x_data}))

    print("The shape of x:\t", x.shape, ",\t and the x.reshape(5,5) is :")

    print(x.reshape(5, 5))

    print("")

    print("The shape of x:\t", W.shape, ",\t and the W.reshape(3,3) is :")

    print(W.reshape(3, 3))

    print("")

    print("The shape of y:\t", y.shape, ",\t and the y.reshape(1) is :")

    print(y.reshape(1))

    print("")
