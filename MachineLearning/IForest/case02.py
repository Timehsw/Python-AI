# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/11/5
    Desc : kaggle案例
    Note : https://www.kaggle.com/mlg-ulb/creditcardfraud/home
    1. Forest
    2. Isolation Tree
    3. Evaluation (Path Length)
'''

# !/usr/bin/python
# -*- coding:utf-8 -*-

##All General Import Statements
import pandas as pd
import numpy as np
import math
import random
import random
from matplotlib import pyplot
import os

df = pd.read_csv("./datas/creditcard_sample2000.csv")
y_true = df['Class']
df_data = df.drop('Class', 1)

sampleSize = 10000

# 降维画图
from sklearn.manifold import TSNE

df_plt = df[df['Class'] == 0].sample(1000)
df_plt_pos = df[df['Class'] == 1].sample(20)
df_plt = pd.concat([df_plt, df_plt_pos])
y_plt = df_plt['Class']
X_plt = df_plt.drop('Class', 1)

X_embedded = TSNE(n_components=2).fit_transform(X_plt)

pyplot.figure(figsize=(12, 8))
pyplot.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_plt, cmap=pyplot.cm.get_cmap("Paired", 2))
pyplot.colorbar(ticks=range(2))

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor  ## Only available with scikit-learn 0.19 and later
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_data, y_true, test_size=0.3, random_state=42)


## Not required for Isolation Forest
def preprocess(df_data):
    for col in df_data:
        df_data[col] = (df_data[col] - np.min(df_data[col])) / (np.max(df_data[col]) - np.min(df_data[col]))
    return


## Not valid for LOF
def train(X, clf, ensembleSize=5, sampleSize=10000):
    mdlLst = []
    for n in range(ensembleSize):
        X = df_data.sample(sampleSize)
        clf.fit(X)
        mdlLst.append(clf)
    return mdlLst


## Not valif for LOF
def predict(X, mdlLst):
    y_pred = np.zeros(X.shape[0])
    for clf in mdlLst:
        y_pred = np.add(y_pred, clf.decision_function(X).reshape(X.shape[0], ))
    y_pred = (y_pred * 1.0) / len(mdlLst)
    return y_pred


from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, f1_score

alg = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01, \
                      max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0, behaviour="new")

if_mdlLst = train(X_train, alg)

if_y_pred = predict(X_test, if_mdlLst)
if_y_pred = 1 - if_y_pred

# Creating class labels based on decision function
if_y_pred_class = if_y_pred.copy()
if_y_pred_class[if_y_pred >= np.percentile(if_y_pred, 95)] = 1
if_y_pred_class[if_y_pred < np.percentile(if_y_pred, 95)] = 0

roc_auc_score(y_test, if_y_pred_class)

f1_score(y_test, if_y_pred_class)

if_cm = confusion_matrix(y_test, if_y_pred_class)

import seaborn as sn

df_cm = pd.DataFrame(if_cm,
                     ['True Normal', 'True Fraud'], ['Pred Normal', 'Pred Fraud'])
pyplot.figure(figsize=(8, 4))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')  # font size
