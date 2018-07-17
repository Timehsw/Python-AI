# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/14
    Desc : 非同型矩阵的加法测试
'''

import numpy as np

arr1 = np.array([
    [2, 3],
    [1, 2]
])

arr2 = np.array([
    [4],
    [5]
])

print(arr1)
print(arr1.shape)
print(arr2)
print(arr2.shape)

arr3 = arr1 + arr2
print(arr3)
print(arr3.shape)

