# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/8
    Desc : 理解梯度下降法例子1
'''

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def f(x):
    return np.power(x,2)


def h(x):
    return 2 * x


step = 0.2
tol = 1e-10
X = []
Y = []

x = 4
y_current = f(x)

X.append(x)
Y.append(y_current)

y_diff = y_current

iter_max = 0

while y_diff > tol and iter_max <= 100:
    x = x - step * h(x)
    tmp = f(x)
    y_diff = y_current - tmp
    y_current = tmp
    X.append(x)
    Y.append(y_current)
    iter_max += 1

print("迭代次数:",iter_max)
print("梯度下降最优解:(%.2f,%.2f)"% (x,f(x)))


# 画图,可视化显示

xs = np.arange(-4.2, 4.2, 0.1)
ys = list(map(lambda l: f(l), xs))

plt.figure(facecolor='w')
plt.title("梯度下降求解最小值")
plt.plot(xs, ys)
plt.plot(X, Y, 'ro-')
plt.xlabel("X")
plt.ylabel('Y')
plt.grid(True)
plt.show()
