# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/24
    Desc :
    Note :
'''

import numpy as np
path = "./datas/arr"

# arr = np.arange(10).reshape(2, 5)

# np.save(path, arr)

arr=np.load(path+'.npy')

print(arr)
