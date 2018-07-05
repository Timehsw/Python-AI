# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/27
    Desc : TF-IDF例子
'''

import warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer, TfidfTransformer

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

arr1 = [
    "This is spark, spark sql a every good",
    "Spark Hadoop Hbase",
    "This is sample",
    "This is anthor example anthor example",
    "spark hbase hadoop spark hive hbase hue oozie",
    "hue oozie spark"
]
arr2 = [
    "this is a sample a example",
    "this c c cd is another another sample example example",
    "spark Hbase hadoop Spark hive hbase"
]
df = arr2

# TF-IDF
# 以TF-IDF的模型来表示文档的特征向量,等价于先做CountVectorizer,然后做TfidTransformer转换操作的结果
tfidf = TfidfVectorizer(min_df=0, dtype=np.float64)
df2 = tfidf.fit_transform(df)
print(df2.toarray())
print(tfidf.get_feature_names())
print(tfidf.get_stop_words())
print('转换另外的文档数据')
print(tfidf.transform(arr1).toarray())

print('~' * 100)
# HashIF-IDF
# 以HashingTF模型来表示文档的特征向量
hashing = HashingVectorizer(n_features=20, non_negative=True, norm=None)
df3 = hashing.fit_transform(df)
print(df3.toarray())
print(hashing.get_stop_words())
print("转换另外的文档数据")
print(hashing.transform(arr1).toarray())

print('~' * 100)
# 以词袋法的形式表示文档
count = CountVectorizer(min_df=0.1, dtype=np.float64, ngram_range=(0, 1))
df4 = count.fit_transform(df)
print(df4.toarray())
print(count.get_stop_words())
print(count.get_feature_names())
print('转换另外的文档数据')
print(count.transform(arr1).toarray())
print(df4)

print('~' * 100)
# 使用改进的TF-IDF公式对文档的特征向量矩阵(数值型的)进行重计算的操作,
tfidf2 = TfidfTransformer()
df5 = tfidf2.fit_transform(df4)
print(df5.toarray())
print('转换另外的文档数据')
print(tfidf2.transform(count.transform(arr1)).toarray())

print('~' * 100)
# 词袋法
dataset = [['my', 'dog', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
           ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
           ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
           ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
           ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
           ['quit', 'worthless', 'buying', 'worthless', 'dog', 'food', 'stupid']]
vocabSet = set()
for doc in dataset:
    vocabSet |= set(doc)
vocabList = list(vocabSet)

SOW = []
for doc in dataset:
    vec = [0]*len(vocabList)
    for i, word in enumerate(vocabList):
        if word in doc:
            vec[i] = 1
    SOW.append(vec)

# 词袋模型
BOW = []
for doc in dataset:
    vec = [0]*len(vocabList)
    for word in doc:
        vec[vocabList.index(word)] += 1
    BOW.append(vec)

print ("词袋法以及词集法")
print (np.array(SOW))
print (np.array(BOW))