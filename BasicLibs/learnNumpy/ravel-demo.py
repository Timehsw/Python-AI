# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/12
    Desc : numpy - ravel 将多维数组降位一维
'''

import numpy as np

arr = np.random.randint(9, size=10).reshape((2, 5))
print(arr)
print(arr.ravel())
