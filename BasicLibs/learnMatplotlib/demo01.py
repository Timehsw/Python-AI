# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/12/15
    Desc : 
    Note : 
'''

import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pydotplus
import numpy as np
import pandas as pd
from sklearn import tree
from xgboost import plot_tree

tpr = np.array([0.87336245, 0.89082969, 0.90393013, 0.98253275, 0.99126638,
                0.99563319, 1., 1.])

fpr = np.array([0., 0.00353982, 0.00707965, 0.03893805, 0.0460177,
                0.05132743, 0.06017699, 1.])

best_threshold = 0.3333333333333333

best_score = 327

ks = tpr - fpr
tmp = np.array(range(1, len(ks) + 1))
tile = 1.0 * tmp / len(tmp)
best_index = pd.Series(ks).idxmax()
ks_value = ks[best_index]
ks_x = tile[best_index]

fig = plt.figure(figsize=[8.4, 6.4])
ax = fig.add_subplot(1, 1, 1)
ax.plot(tile, tpr, color='green', label='TPR', linestyle='-')
ax.plot(tile, fpr, color='red', label='FPR', linestyle='-')
ax.plot(tile, ks, color='blue', label='ks', linestyle='-')
ax.axhline(ks_value, color='gray', linestyle='--')
ax.axvline(ks_x, color='gray', linestyle='--')
ax.set_title('best_threshold : {}    ks : {}    best_score : {}'.format(round(best_threshold, 4), round(ks_value, 4),
                                                                        best_score))
ax.set_yticks(np.arange(0, 1, 0.1))
plt.axis('equal')
plt.legend()
plt.show()
