# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/8
    Desc : 分布式客户端
'''

import tensorflow as tf
import numpy as np

# 1. 构建图
with tf.device('/job:ps/task:0'):
    # 2. 构造数据
    x = tf.constant(np.random.rand(100).astype(np.float32))

# 3. 使用另外一个机器
with tf.device('/job:work/task:1'):
    y = x * 0.1 + 0.3

# 4. 运行
with tf.Session(target='grpc://127.0.0.1:33335',
                config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    print(sess.run(y))

