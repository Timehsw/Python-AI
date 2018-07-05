# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/2
    Desc : KNN算法用于邮件分类
'''

import numpy as np
import pandas as pd
import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score

df = pd.read_csv('./datas/result_process02.csv')
df.dropna(axis=0, how='any', inplace=True)
print(df.columns.tolist())
print(df.head())

# 训练集测试集划分
x_train, x_test, y_train, y_test = train_test_split(df[['has_date', 'jieba_cut_content', 'content_length_sema']],
                                                    df['label'], test_size=0.2, random_state=0)

print('训练集大小:', x_train.shape)
print('测试集大小:', x_test.shape)

print('-' * 20, '开始训练集的特征工程', '-' * 20)
transformer = TfidfVectorizer(norm='l2', use_idf=True)
svd = TruncatedSVD(n_components=20)
jieba_cut_content = list(x_train['jieba_cut_content'].astype('str'))
transformer_model = transformer.fit(jieba_cut_content)
df1 = transformer_model.transform(jieba_cut_content)
svd_model = svd.fit(df1)
df2 = svd_model.transform(df1)
data = pd.DataFrame(df2)
print(data.head(10))
print(data.info())

data['has_date'] = list(x_train['has_date'])
data['content_length_sema'] = list(x_train['content_length_sema'])

print('-' * 20, 'KNN', '-' * 20)

knn = KNeighborsClassifier(n_neighbors=5)
model = knn.fit(data, y_train)

print('-' * 20, '构建测试集', '-' * 20)
jieba_cut_content_test = list(x_test['jieba_cut_content'].astype('str'))
data_test = pd.DataFrame(svd_model.transform(transformer_model.transform(jieba_cut_content_test)))

data_test['has_date'] = list(x_test['has_date'])
data_test['content_length_sema'] = list(x_test['content_length_sema'])

print('-' * 20, 'knn预测', '-' * 20)
y_predict = model.predict(data_test)

precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1mean = f1_score(y_test, y_predict)

print('精确率为: %0.5f' % precision)
print('召回率为: %0.5f' % recall)
print('F1平均值为: %0.5f' % f1mean)
