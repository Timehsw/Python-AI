# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-08-02
    Desc : xgb调参
    Note : 
'''

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
from sklearn import metrics
from sklearn.datasets import make_hastie_10_2
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import pandas as pd

def model_cv(model,X,y,cv_folds=5,early_stopping_rounds=50,seed=0):
    xgb_param = model.get_xgb_params()
    xgtrain = xgb.DMatrix(X,label=y)
    cvresult = xgb.cv(xgb_param,xgtrain,num_boost_round=model.get_params()['n_estimators'],nfold=cv_folds,
                      metrics='auc',seed=seed,callbacks=[
            xgb.callback.print_evaluation(show_stdv=False),
            xgb.callback.early_stop(early_stopping_rounds)
        ])
    num_round_best = cvresult.shape[0]-1
    print("Best round num: ",num_round_best)
    return num_round_best


# 准备数据，y本来是[-1:1],xgboost自带接口邀请标签是[0:1],把-1的转成1了。
X, y = make_hastie_10_2(random_state=0)
X = DataFrame(X)
y = DataFrame(y)
y.columns = {"label"}
label = {-1: 0, 1: 1}
y.label = y.label.map(label)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 划分数据集

# 第一步，从此参数中，利用xgb.cv选出最优的n_estimators

n_estimators = 5000
seed = 0
max_depth = 3
min_child_weight = 7
gamma = 0
subsample = 0.8
colsample_bytree = 0.8
scale_pos_weight = 1
reg_alpha = 1
reg_lambda = 1e-5
learning_rate = 0.1
model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                      min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, reg_alpha=reg_alpha,
                      reg_lambda=reg_lambda, colsample_bytree=colsample_bytree, objective='binary:logistic',
                      nthread=4, scale_pos_weight=scale_pos_weight, seed=seed)
num_round = model_cv(model, X_train, y_train)
# 602
# 在开始的时候，可以选择较大一点的 learning_rate，这样可以更快地收敛，计算出最佳的迭代次数。
# 然后，使用 Sklearn 的 GridSearchCV 自动测试参数。


# 第二步：利用sklearn 中的gridsearchcv寻找最优参数组合

def gridsearch_cv(model,test_param,X,y,cv=5):
    gsearch = GridSearchCV(estimator=model,param_grid=test_param,scoring='roc_auc',n_jobs=4,iid=False,cv=cv)
    gsearch.fit(X,y)
    print("CV Results: ",gsearch.cv_results_)
    print("Best Params: ",gsearch.best_params_)
    print("Best Score: ",gsearch.best_score_)
    return gsearch.best_params_

# 提供模型及候选参数的列表，GridSearchCV 能自动穷举所有组合的参数，计算最佳的参数组合。
# 首先调试 max_depth 和 min_child_weight 参数组合。

# tune max_depth & min_child_weight
param_test1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 10, 2)
}
gridsearch_cv(model, param_test1, X_train,X_train)

# 函数能从所有的候选组合中选出误差最小的组合，注意开始的时候不要给太多组合，不然计算会非常慢。可以先给出一个大范围，然后在慢慢缩小范围。
# 例如，如果 max_depth = 3，min_child_weight = 1 时最佳，则缩小范围再试一次。

# 然后调整 gamma。
# tune gamma
param_test2 = {
    'gamma': [i / 10.0 for i in range(0, 5)]
}
gridsearch_cv(model, param_test2, X_train, X_train)

# 接着类似调整其他参数，先从大范围开始，慢慢缩小范围。

# tune subsample & colsample_bytree
param_test3 = {
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)]
}
gridsearch_cv(model, param_test3, X_train,X_train)
# tune scale_pos_weight
param_test4 = {
    'scale_pos_weight': [i for i in range(1, 10, 2)]
}
gridsearch_cv(model, param_test4, X_train,X_train)
# tune reg_alpha
param_test5 = {
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100, 1000]
}
gridsearch_cv(model, param_test5, X_train,X_train)
# tune reg_lambda
param_test6 = {
    'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100, 1000]
}
gridsearch_cv(model, param_test6, X_train,X_train)



# #XGBoost自带接口


# params={
#     'eta': 0.3, # 学习率参数
#     'max_depth':3,
#     'min_child_weight':1,
#     'gamma':0.3,
#     'subsample':0.8,
#     'colsample_bytree':0.8,
#     'booster':'gbtree',
#     'objective': 'binary:logistic',
#     'nthread':12,
#     'scale_pos_weight': 1,
#     'lambda':1,
#     'seed':27,
#     'silent':1 ,
#     'eval_metric': 'auc'
# }
# d_train = xgb.DMatrix(X_train, label=y_train)
# d_valid = xgb.DMatrix(X_test, label=y_test)
# d_test = xgb.DMatrix(X_test)
# watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# print("XGBoost_自带接口进行训练：")
# model_bst = xgb.train(params, d_train, 30, watchlist, early_stopping_rounds=500, verbose_eval=10)
#
# y_bst = model_bst.predict(d_test)
#
# # 第三步：评估结果
# print("XGBoost_自带接口    AUC Score : %f" % metrics.roc_auc_score(y_test, y_bst))
#
# # 将概率值转化为0和1
# y_bst = pd.DataFrame(y_bst).apply(lambda row: 1 if row[0] >= 0.5 else 0, axis=1)
# print("XGBoost_自带接口    AUC Score : %f" % metrics.accuracy_score(y_test, y_bst))
# '''
# XGBoost_自带接口    AUC Score : 0.970292
# XGBoost_sklearn接口 AUC Score : 0.970292
# XGBoost_自带接口    AUC Score : 0.897917
# XGBoost_sklearn接口 AUC Score : 0.897917
# '''
#
# # 模型导出功能
# model_bst.dump_model("./rawxgb.txt")
# clf.get_booster().dump_model("./skxgb.txt")