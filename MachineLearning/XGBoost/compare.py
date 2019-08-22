# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-08-02
    Desc : 
    Note : 
'''

from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sklearn import metrics
from sklearn.datasets import make_hastie_10_2
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import pandas as pd

# 准备数据，y本来是[-1:1],xgboost自带接口邀请标签是[0:1],把-1的转成1了。
X, y = make_hastie_10_2(random_state=0)
X = DataFrame(X)
y = DataFrame(y)
y.columns = {"label"}
label = {-1: 0, 1: 1}
y.label = y.label.map(label)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 划分数据集

# 第二步：分别使用两个接口进行训练和预测。两种接口的参数完全一样。

#XGBoost自带接口
params={
    'eta': 0.3,
    'max_depth':3,
    'min_child_weight':1,
    'gamma':0.3,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'nthread':12,
    'scale_pos_weight': 1,
    'lambda':1,
    'seed':27,
    'silent':0 ,
    'eval_metric': 'auc'
}
d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_test, label=y_test)
d_test = xgb.DMatrix(X_test)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

# sklearn接口
clf = XGBClassifier(
    n_estimators=30,  # 三十棵树
    learning_rate=0.3,
    max_depth=3,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=27)

print("XGBoost_自带接口进行训练：")
model_bst = xgb.train(params, d_train, 30, watchlist, early_stopping_rounds=500, verbose_eval=10)
print("XGBoost_sklearn接口进行训练：")
model_sklearn = clf.fit(X_train, y_train)

y_bst = model_bst.predict(d_test)
y_sklearn = clf.predict_proba(X_test)[:, 1]

# 第三步：评估结果
print("XGBoost_自带接口    AUC Score : %f" % metrics.roc_auc_score(y_test, y_bst))
print("XGBoost_sklearn接口 AUC Score : %f" % metrics.roc_auc_score(y_test, y_sklearn))

# 将概率值转化为0和1
y_bst = pd.DataFrame(y_bst).apply(lambda row: 1 if row[0] >= 0.5 else 0, axis=1)
y_sklearn = pd.DataFrame(y_sklearn).apply(lambda row: 1 if row[0] >= 0.5 else 0, axis=1)
print("XGBoost_自带接口    AUC Score : %f" % metrics.accuracy_score(y_test, y_bst))
print("XGBoost_sklearn接口 AUC Score : %f" % metrics.accuracy_score(y_test, y_sklearn))
'''
XGBoost_自带接口    AUC Score : 0.970292
XGBoost_sklearn接口 AUC Score : 0.970292
XGBoost_自带接口    AUC Score : 0.897917
XGBoost_sklearn接口 AUC Score : 0.897917
'''

# 模型导出功能
model_bst.dump_model("./rawxgb.txt")
clf.get_booster().dump_model("./skxgb.txt")