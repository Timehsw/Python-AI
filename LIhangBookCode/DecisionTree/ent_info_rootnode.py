# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/12/19
    Desc : 
    Note : 李航-统计学习方法 决策树算法 ,根据信息增益计算根节点
'''
import numpy as np
import pandas as pd
from math import log
from LIhangBookCode.DecisionTree.dataset import create_data


def calc_ent(datasets):
    '''
    计算熵
    :param datasets:
    :return:
    '''
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    ent = -sum([(p / data_length) * log(p / data_length, 2) for p in label_count.values()])
    return ent


def cond_ent(datasets, axis=0):
    '''
    计算经验条件熵
    :param dataset:
    :param axis:
    :return:
    '''
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p) / data_length) * calc_ent(p) for p in feature_sets.values()])
    return cond_ent


def info_gain(ent, cond_ent):
    '''
    计算信息增益
    :param ent:
    :param cond_ent:
    :return:
    '''
    return ent - cond_ent


def info_gain_train(datasets):
    '''
    计算数据集各个特征的信息增益
    :param datasets:
    :return:
    '''
    # 特征个数
    count = len(datasets[0]) - 1
    # 原始数据集的信息熵
    ent = calc_ent(datasets)
    best_feature = []
    for c in range(count):
        c_info_gain = info_gain(ent, cond_ent(datasets, axis=c))
        best_feature.append((c, c_info_gain))
        print('特征({}) - info_gain - {:.3f}'.format(labels[c], c_info_gain))

    best_ = max(best_feature, key=lambda x: x[-1])
    return '特征({})的信息增益最大,选择为根节点特征'.format(labels[best_[0]])


datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns=labels)
print(train_data.head())

print(info_gain_train(np.array(datasets)))
