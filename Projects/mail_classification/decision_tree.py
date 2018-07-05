# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/2
    Desc : 决策树用于邮件分类
'''

import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

path = "./datas/result_process02.csv"
df = pd.read_csv(path, sep=',')
df.dropna(axis=0, how='any', inplace=True)
print(df.head())
print(df.columns.tolist())

x_train, x_test, y_train, y_test = train_test_split(df[['has_date', 'jieba_cut_content', 'content_length_sema']],
                                                    df['label'], test_size=0.2, random_state=0)

print('训练数据集大小: ', x_train.shape)
print('测试数据集大小: ', x_test.shape)

print('-' * 30, '开始计算tf-idf', '-' * 30)
jieba_cut_content = list(x_train['jieba_cut_content'].astype('str'))
transformer = TfidfVectorizer(norm='l2', use_idf=True)
transformer_model = transformer.fit(jieba_cut_content)
df1 = transformer_model.transform(jieba_cut_content)

print('-' * 30, '开始SVD分解降维', '-' * 30)
svd = TruncatedSVD(n_components=20)
svd_model = svd.fit(df1)
df2 = svd_model.transform(df1)
data = pd.DataFrame(df2)

print('-' * 30, '开始重新构建矩阵', '-' * 30)
data['has_date'] = list(x_train['has_date'])
data['content_length_sema'] = list(x_train['content_length_sema'])

print('-' * 30, '构建决策树模型', '-' * 30)
tree = DecisionTreeClassifier(criterion='gini', max_depth = 5, random_state = 0)
model = tree.fit(data, y_train)

print('-' * 30, "构建测试集", '-' * 30)
jieba_cut_content_test = list(x_test['jieba_cut_content'].astype('str'))
data_test = pd.DataFrame(svd_model.transform(transformer_model.transform(jieba_cut_content_test)))
data_test['has_date'] = list(x_test['has_date'])
data_test['content_length_sema'] = list(x_test['content_length_sema'])

print('-' * 30, "开始测试集预测", '-' * 30)
start = time.time()
y_predict = model.predict(data_test)
end = time.time()
print('预测共耗时%.2fs' % (end - start))

print('-' * 30, "开始评估预测模型", '-' * 30)
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1mean = f1_score(y_test, y_predict)

print('-' * 30, "开始输出预测结果", '-' * 30)
print("精确率为: %0.5f" % precision)
print("召回率: %0.5f" % recall)
print("F1均值为: %0.5f" % f1mean)
