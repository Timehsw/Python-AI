# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/11/16
    Desc : 
    Note : 
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA  # 主成分分析
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier, XGBModel

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


path = '/Users/hushiwei/GEO/数据集/建模数据集/cleand_1416c.csv'

df = pd.read_csv(path)
y = df['y_label']
raw_x = df.drop(['y_label'], axis=1)
# apply_time
print('--------------')

# 解析时间
raw_x['apply_time'] = pd.to_datetime(raw_x['apply_time'])
# 提取月份出来
raw_x['apply_month'] = raw_x['apply_time'].dt.month
raw_x['apply_year'] = raw_x['apply_time'].dt.year

# 研究object类型中是否有比较有价值的特征
raw_x.select_dtypes(include=['object']).head()

# 方差选择法
# 从数值类型中做一个方差最大的特征选择
# threshold各个特征属性的阈值,获取方差大于阈值的特征
variance = VarianceThreshold(threshold=0.8)
# (30000, 1353)
df_choice_float = variance.fit_transform(raw_x.select_dtypes(include=['float64']))
# (30000, 350)

# 再做一个卡方检验

# x=raw_x.select_dtypes(include=['float64','int64']).head()
x=raw_x.iloc[:,3:40]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)

## 数据正则化操作(归一化)
ss = StandardScaler()
## 模型训练一定是在训练集合上训练的
X_train = ss.fit_transform(X_train)  ## 训练正则化模型，并将训练数据归一化操作
X_test = ss.transform(X_test)  ## 使用训练好的模型对测试数据进行归一化操作

## Logistic算法模型构建
# LogisticRegression中，参数说明：
# penalty => 惩罚项方式，即使用何种方式进行正则化操作(可选: l1或者l2)
# C => 惩罚项系数，即L1或者L2正则化项中给定的那个λ系数(ppt上)
# LogisticRegressionCV中，参数说明：
# LogisticRegressionCV表示LogisticRegression进行交叉验证选择超参数(惩罚项系数C/λ)
# Cs => 表示惩罚项系数的可选范围
lr = LogisticRegressionCV(Cs=np.logspace(-4, 1, 50), fit_intercept=True, penalty='l2', solver='lbfgs', tol=0.01,
                          multi_class='ovr')
lr.fit(X_train, Y_train)
y_train_pre=lr.predict(X_train)

## Logistic算法效果输出
lr_r = lr.score(X_train, Y_train)
print("Logistic算法R值（训练集上的准确率）：", lr_r)
# print("Logistic算法稀疏化特征比率：%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100))
# print("Logistic算法参数：", lr.coef_)
# print("Logistic算法截距：", lr.intercept_)

## Logistic算法预测（预测所属类别）
lr_y_predict = lr.predict(X_test)
print("测试集上的预测值", lr.score(X_test, Y_test))

## Logistic算法获取概率值(就是Logistic算法计算出来的结果值)
# y1 = lr.predict_proba(X_test)

# 计算ks值
fpr, tpr, thresholds = roc_curve(Y_train, y_train_pre)
r = tpr - fpr
ks = np.max(r)
print('训练集的ks : ', ks)

fpr, tpr, thresholds = roc_curve(Y_test, lr_y_predict)
r = tpr - fpr
ks = np.max(r)
print('测试集的ks : ', ks)

print(roc_auc_score(Y_test,lr_y_predict))


# 选出列名中是近6个月的
# df3[df3['name'].str.endswith('M6')]