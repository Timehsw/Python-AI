# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/18
    Desc : 
'''

import numpy as np
import matplotlib.pyplot as plt

image = np.random.uniform(0, 255, 300).reshape((10, 10, 3))
print(image.shape)
print(image)

plt.imshow(image)
plt.show()
