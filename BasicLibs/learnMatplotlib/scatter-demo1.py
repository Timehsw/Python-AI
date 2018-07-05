# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/6
    Desc : 
'''

import numpy as np
import matplotlib.pyplot as plt

# 产生测试数据
x = np.arange(-10, 10)
y=list(map(lambda l:l*l,x))

print(x)
print(y)
print('~'*100)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('Scatter plot demo1')
plt.xlabel('X')
plt.ylabel('Y')
ax.scatter(x,y,c='r',marker='x')

plt.grid(True)
plt.show()
