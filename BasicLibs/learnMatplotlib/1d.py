# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/8
    Desc : matplotlib画一维图像
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 一维原始图像
def f1(x):
    return 0.5 * (x - 0.25) ** 2

# 构建数据
X = np.arange(-4, 4.5, 0.05)
Y = np.array(list(map(lambda t: f1(t), X)))

# 画图
plt.figure(facecolor='w')
plt.plot(X, Y, 'r-', linewidth=2)
plt.title(u'函数$y=0.5 * (θ - 0.25)^2$')
plt.show()
