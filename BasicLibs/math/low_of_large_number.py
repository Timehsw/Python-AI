# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/4/1.

    desc: 大数定理展示理解
'''

import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 解决中文显示问题
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

random.seed(28)


def generate_random_int(n):
    '''
    产生n个1-9的随机数
    :param n:
    :return:
    '''
    return [random.randint(1,9) for i in range(n)]

if __name__ == '__main__':
    number=8000
    x=[i for i in range(number+1) if i!=0]

    # 产生number个[1,9]的随机数
    total_random_int=generate_random_int(number)

    # 求n个[1,9]的随机数的均值,n=1,2,3,4,5...
    # 求这number个数的均值,比如1个元素的均值,2个元素的均值,3个元素的均值,4个元素的均值,一直到最后
    y=[np.mean(total_random_int[0:i+1]) for i in range(number)]


    plt.plot(x,y,'b-')
    plt.xlim(0,number)
    plt.grid(True)
    plt.show()