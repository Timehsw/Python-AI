# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/19
    Desc : 案例一/EM分类初始及GMM算法实现
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import multivariate_normal  # 多元正态分布
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_distances_argmin

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 创建模拟数据(3维数据)
np.random.seed(28)
N = 500
M = 500

# 根据给定的均值和协方差矩阵构建数据
mean1 = (0, 0, 0)
cov1 = np.diag((1, 2, 3))
# 产生400条数据
data1 = np.random.multivariate_normal(mean1, cov1, N)

# 产生一个数据分布不均衡的数据集,100条
mean2 = (2, 2, 1)
cov2 = np.array(((3, 1, 0), (1, 3, 0), (0, 0, 3)))
data2 = np.random.multivariate_normal(mean2, cov2, M)

# 合并data1和data2这两个数据集
data = np.vstack((data1, data2))

# 产生数据对应的y值
y1 = np.array([True] * N + [False] * M)
y2 = ~y1

# 预测结果(得到概率密度值)
style = 'sklearn'
# style='self'
