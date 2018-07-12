# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/8
    Desc : 理解梯度下降法例子2
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


## 原函数
def f(x):
    return x ** 2


## 导数
def h(x):
    return 2 * x


X = []
Y = []

x = 2
step = 0.8
f_change = f(x)
f_current = f(x)
X.append(x)
Y.append(f_current)
while f_change > 1e-10:
    x = x - step * h(x)
    tmp = f(x)
    f_change = np.abs(f_current - tmp)
    f_current = tmp
    X.append(x)
    Y.append(f_current)
print('最终结果为：', (x, f_current))

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False

fig = plt.figure()
X2 = np.arange(-2.1, 2.15, 0.05)
Y2 = X2 ** 2

plt.plot(X2, Y2, '-', color='#666666', linewidth=2)
plt.plot(X, Y, 'bo--')
plt.title('$y=x^2$函数求解最小值，最终解为：x=%.2f,y=%.2f' % (x, f_current))
plt.show()
