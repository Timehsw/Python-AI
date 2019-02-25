# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-02-15
    Desc : 
    Note : 
'''

import json
import numpy as np
import pandas as pd

file = open(r"/Users/hushiwei/IdeaProjects/ConvertLightGBM/src/main/resources/gbm_has_missing.json", "rb")  # 读取模型json文件
# file = open(r"/Users/hushiwei/IdeaProjects/ConvertLightGBM/src/main/resources/gbm.json", "rb")  # 读取模型json文件
# file = open(r"/Users/hushiwei/Downloads/model2/model/gbm.json", "rb")  # 读取模型json文件
model = json.load(file)

feature_names = model['feature_names']  # 获取模型中所用的特征变量


# 定义一个函数判断每一个leaf是走left还是right
def decison(data,threshold,default_left):
    '''
    :param data:  特征值
    :param threshold: 分割判断值
    :param default_left: 默认分支 default_left= True or False
    :return: 返回结果left_child or right_child
    '''
    if ((np.isnan(data)) and (default_left is True)):
        return 'left_child'
    elif data <= threshold:
        return 'left_child'
    else:
        return 'right_child'

# 定义预测函数
def predict_gbm(data):
    score = 0
    for i in range(len(model['tree_info'])):  # 遍历每一个节点
        num_leaves = model['tree_info'][i]['num_leaves']  # 获取每颗树的节点数
        tree = model['tree_info'][i]['tree_structure']  # 获取每一颗树结构
        for i in range(num_leaves):  # 遍历节点数
            # 到达节点leaf,进行走向判断
            threshold = tree.get('threshold')
            default_left = tree.get('default_left')
            split_feature = feature_names[tree['split_feature']]  # 获取叶子节点的分割特征变量
            next_decison = decison(data[split_feature], threshold, default_left)
            # 获取下一个分支leaf
            tree = tree[next_decison]
            if tree.get('left_child', 'not found') == 'not found':  # 如果到达节点节点停止遍历，返回对应值
                score = score + tree['leaf_value']
                break
    return (score)


# 进行测试
input = "/Users/hushiwei/IdeaProjects/ConvertLightGBM/src/main/resources/test.csv"
# input = "/Users/hushiwei/IdeaProjects/ConvertLightGBM/src/main/resources/test_woe.csv"
# input = "/Users/hushiwei/IdeaProjects/ConvertLightGBM/src/main/resources/test_no_missing.csv"
df = pd.read_csv(input)
df = df.iloc[:, 1:]
predict_df = []
for i in range(len(df)):
    predict_data = predict_gbm(df.iloc[i, :])  # 分值
    predict_dt = 1 / (np.exp(-predict_data) + 1)  # 将预测分值转为p值
    predict_df.append(predict_dt)

print(predict_df)
