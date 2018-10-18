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

n = 3  # 设置柱子数
width = 0.5 / n

left = np.arange(1, 5)
height = np.array([200, 300, 400, 500])

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.bar(left - width, height, width=width, color='g', align='center', label='green')
ax.bar(left, height * 3, width=width, color='y', align='center', label='yellow')
ax.bar(left + width, height * 2, width=width, color='b', align='center', label='blue')

ax.set_xticks(np.arange(1, 5))

ax.set_xticklabels([str(i) + '月' for i in np.arange(1, 6)],rotation=30,fontsize='small')

ax.set_xlabel('月份')
ax.set_ylabel('美元')
ax.legend()
ax.set_title('bat pic 2')
plt.show()
