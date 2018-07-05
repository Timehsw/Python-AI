# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/8
    Desc : matplotlib画2维图像
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 二维原始图像

def f(x, y):
    return 0.6 * (x + y) ** 2 - x * y


# 构建数据
X1 = np.arange(-2, 2.2, 0.5)
X2 = np.arange(-2, 2.2, 0.5)
print('X1',X1)
print('X2',X2)

print('~'*100)

X1, X2 = np.meshgrid(X1, X2)
Y = np.array(list(map(lambda t: f(t[0], t[1]), zip(X1.flatten(), X2.flatten()))))

print('Y--',Y)
print('Y--',Y.shape)
Y.shape = X1.shape

print('~'*100)
print('X1',X1)
print('X2',X2)
print('Y',Y)

# 画图
fig = plt.figure(facecolor='w')
ax = Axes3D(fig)
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=plt.cm.jet)
ax.set_title(u'函数$y=0.6 * (θ1 + θ2)^2 - θ1 * θ2$')
plt.show()
