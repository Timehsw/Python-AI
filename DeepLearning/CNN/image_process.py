# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/18
    Desc : 图像处理的python库:opencv,PIL,matplotlib,tensorflow等
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 打印numpy的数组对象的时候,中间不省略
np.set_printoptions(threshold=np.inf)


def show_image_tensor(image_tensor):
    # 要求:使用交互式会话
    # 获取图像tensor对象对应的image对象,Image对象是一个[h,w,c]
    image = image_tensor.eval()
    print('图像大小为:{}', format(image.shape))

    if len(image.shape) == 3 and image.shape[2] == 1:
        # 黑白图像
        plt.imshow(image[:, :, 0], camp='Greys_r')
        plt.show()
    elif len(image.shape) == 3:
        # 彩色图像
        plt.imshow(image)
        plt.show()


# 1. 交互式会话启动
session = tf.InteractiveSession()

image_path='./data'
# 一,图像格式的转换
