# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/18
    Desc : 柱状图
    Note : 
'''

'''
matplotlib.pyplot. bar (*args, **kwargs) bar(left, height, width, bottom, * args, align='center', **kwargs) 参数： left:数据标量 height：高 width:款 bottom：底端对应Y轴align:对齐如果为 "居中", 则将x参数解释为条形中心的坐标。如果 "边缘", 将条形按其左边缘对齐要对齐右边缘的条形图, 可传递负的宽度和对align='edge'

'''
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar([1, 3, 5, 7, 9, 11], [5, 2, 7, 8, 2, 6], label='Example One', color='y')
ax.bar([2, 4, 6, 8, 10, 12], [8, 6, 2, 5, 6, 3], label='Example Two', color='g')

plt.legend()
plt.xlabel('bar number')
plt.ylabel('bar height')
plt.title('bat pic')
plt.show()
