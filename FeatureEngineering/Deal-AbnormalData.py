# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/28
    Desc : 异常数据处理
'''

import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

path1='datas/C0904.csv'
path2='datas/C0911.csv'
filename = "X12CO2" # H2O

'''
### 原始数据读取
plt.figure(figsize=(10, 6), facecolor='w')
plt.subplot(121)
data = pd.read_csv(path1, header=0)
x = data[filename].values
plt.plot(x, 'r-', lw=1, label=u'C0904')
plt.title(u'实际数据0904', fontsize=18)
plt.legend(loc='upper right')
plt.xlim(0, 80000)
plt.grid(b=True)

plt.subplot(122)
data = pd.read_csv(path2, header=0)
x = data[filename].values
plt.plot(x, 'r-', lw=1, label=u'C0911')
plt.title(u'实际数据0911', fontsize=18)
plt.legend(loc='upper right')
plt.xlim(0, 80000)
plt.grid(b=True)

plt.tight_layout(2, rect=(0, 0, 1, 0.95))
plt.suptitle(u'如何找到下图中的异常值', fontsize=20)
plt.show()

'''


### 异常数据处理
data = pd.read_csv(path2, header=0)
x = data[filename].values

width = 300
delta = 5
eps = 0.02
N = len(x)
p = []
# 异常值存储
abnormal = []
for i in np.arange(0, N-width, delta):
    s = x[i:i+width]
    ## 获取max-min的差值
    min_s = np.min(s)
    ptp = np.ptp(s)
    ptp_min = ptp / min_s
    p.append(ptp_min)
    ## 如果差值大于给定的阈值认为是
    if ptp_min > eps:
        abnormal.append(range(i, i+width))
## 获得异常的数据x值
abnormal = np.array(abnormal).flatten()
abnormal = np.unique(abnormal)
#plt.plot(p, lw=1)
#plt.grid(b=True)
#plt.show()

plt.figure(figsize=(18, 7), facecolor='w')
plt.subplot(131)
plt.plot(x, 'r-', lw=1, label=u'原始数据')
plt.title(u'实际数据', fontsize=18)
plt.legend(loc='upper right')
plt.xlim(0, 80000)
plt.grid(b=True)

plt.subplot(132)
t = np.arange(N)
plt.plot(t, x, 'r-', lw=1, label=u'原始数据')
plt.plot(abnormal, x[abnormal], 'go', markeredgecolor='g', ms=3, label=u'异常值')
plt.legend(loc='upper right')
plt.title(u'异常数据检测', fontsize=18)
plt.xlim(0, 80000)
plt.grid(b=True)

# 预测
plt.subplot(133)
select = np.ones(N, dtype=np.bool)
select[abnormal] = False
t = np.arange(N)
## 决策树
dtr = DecisionTreeRegressor(criterion='mse', max_depth=10)
br = BaggingRegressor(dtr, n_estimators=10, max_samples=0.3)
## 模型训练
br.fit(t[select].reshape(-1, 1), x[select])
## 模型预测得出结果
y = br.predict(np.arange(N).reshape(-1, 1))
y[select] = x[select]
plt.plot(x, 'g--', lw=1, label=u'原始值')    # 原始值
plt.plot(y, 'r-', lw=1, label=u'校正值')     # 校正值
plt.legend(loc='upper right')
plt.title(u'异常值校正', fontsize=18)
plt.xlim(0, 80000)
plt.grid(b=True)

plt.tight_layout(1.5, rect=(0, 0, 1, 0.95))
plt.suptitle(u'异常值检测与校正', fontsize=22)
plt.show()

