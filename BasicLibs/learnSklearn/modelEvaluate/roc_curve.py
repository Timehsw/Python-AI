# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/9/12
    Desc : 
    Note : 
'''
import numpy as np
from sklearn.metrics import roc_curve

'''
ROC曲线
TPR:真正例率 为纵坐标
FPR:假正例率 为横坐标
绘制曲线

纵坐标：真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
TPR = TP /（TP + FN）  （正样本预测结果数 / 正样本实际数）

横坐标：假正率（False Positive Rate , FPR）
FPR = FP /（FP + TN） （被预测为正的负样本结果数 /负样本实际数）

fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)
该函数返回这三个变量：fpr,tpr,和阈值thresholds;
理解thresholds:
分类器的一个重要功能“概率输出”，即表示分类器认为某个样本具有多大的概率属于正样本（或负样本）

接下来，我们从高到低，依次将“Score”值作为阈值threshold，当测试样本属于正样本的概率大于或等于这个threshold时，我们认为它为正样本，否则为负样本。每次选取一个不同的threshold，我们就可以得到一组FPR和TPR，即ROC曲线上的一点。当我们将threshold设置为1和0时，分别可以得到ROC曲线上的(0,0)和(1,1)两个点。将这些(FPR,TPR)对连接起来，就得到了ROC曲线。当threshold取值越多，ROC曲线越平滑。其实，我们并不一定要得到每个测试样本是正样本的概率值，只要得到这个分类器对该测试样本的“评分值”即可（评分值并不一定在(0,1)区间）。评分越高，表示分类器越肯定地认为这个测试样本是正样本，而且同时使用各个评分值作为threshold。我认为将评分值转化为概率更易于理解一些
'''


def printR(name, res):
    print('~' * 10, name, '~' * 10)
    print(res)


y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)
printR('fpr', fpr)
printR('tpr', tpr)
printR('thresholds', thresholds)

from sklearn.metrics import auc

auc = auc(fpr, tpr)
printR('auc', auc)
