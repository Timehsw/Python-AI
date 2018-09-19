# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/9/12
    Desc : 准确率
    Note : 
'''

import numpy as np
from sklearn.metrics import *


def show(name, res):
    print('~' * 10, name, '~' * 10)
    print(res)


y_pred = [0, 2, 1, 3, 9, 9, 8, 5, 8]
y_true = [0, 1, 2, 3, 2, 6, 3, 5, 9]

score = accuracy_score(y_true, y_pred)
show('accuracy_score', score)

score2 = accuracy_score(y_true, y_pred, normalize=False)
show('accuracy_score : normalize=False', score2)

recall_score_micro = recall_score(y_true, y_pred, average='micro')
show('recall_score_micro', recall_score_micro)

recall_score_macro = recall_score(y_true, y_pred, average='macro')
show('recall_score_macro', recall_score_macro)

from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]

res = confusion_matrix(y_true, y_pred)
print(res)
