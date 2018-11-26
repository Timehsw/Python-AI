# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/11/22
    Desc : 金融风控建模
    Note : 参考: https://github.com/ParagPande/Classifier_to_assess_H1B_dataset/blob/3bc20b9a5f/H1B-Visa_predictor_XGBoost.py
    1. 查看每一列数据的空值情况,删除全为空的列;
    2. 将含有时间的列抽取出来,根据业务来安装时间划分,做衍生变量;
    3. 将字符类型的列抽取出来,进行分类做特征;
    4. 将所有的数值型特征抽取出来,通过计算重要度筛选出一部分特征来;
    5. 最后将抽取出来的特征加上衍生变量,入模xgboost,进行模型训练.
'''

import numpy as np
import pandas as pd
import operator
import warnings
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA  # 主成分分析
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier, XGBModel

warnings.filterwarnings("ignore")


def ks_statistic(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true.values, y_pred)
    r = tpr - fpr
    ks = np.max(r)
    return ks


# 读取数据 (30000, 4143)
path = '/Users/hushiwei/GEO/数据集/建模数据集/finace_risk_4000c_181115.csv'
raw_data_df = pd.read_csv(path)

# 将原始数据中的\N字符替换为NaN,以便后续进行缺失值填充
raw_data_df = raw_data_df.replace('\\N', np.NaN)

# 对原始数据中的列,进行缺失值的分布统计
# 统计每一列的缺失值个数
missing_value_count_by_columns = raw_data_df.shape[0] - raw_data_df.count()
missing_value_count_by_columns = missing_value_count_by_columns.map(lambda x: x / raw_data_df.shape[0])

# 取出全为空的列
usefule_missing = missing_value_count_by_columns[missing_value_count_by_columns == 1.0]
usefule_columns = usefule_missing.index.values.tolist()

# 从原始数据中删除全为空的列
raw_data_df.drop(usefule_columns, axis=1, inplace=True)

# 查看原始数据的数据分布 (30000, 2054)
print(raw_data_df.dtypes.value_counts())
# 根据类型来取出数据,进行数据分析
# 只剩下object(2052列)和int64(2列)
# int64只有两列了,一列id,一列是y_laber
raw_data_df.select_dtypes(include=['int64'])

# 取出y列值
Y = raw_data_df.select_dtypes(include=['int64'])['y_label']

# 问题在,object类型中大多数实际上是数值类型,只有少数是类别类型,我需要从中抽取出可以转成数值类型的特征来
tmp_object_df = raw_data_df.select_dtypes(include=['object'])

# 将能转成数值的转成数值类型,如果某一列中有无法转的元素,那么这个列都不转了,保留,可能就是剩下的一些类别列,和时间列了
tmp_object_df = tmp_object_df.apply(pd.to_numeric, errors='ignore')
# 全数值列 (30000, 1854)   ------------> 数值类型列
raw_numeric_cols_df = tmp_object_df.select_dtypes(include=['float64'])

# (30000, 198)
raw_objs_others_cols_df = tmp_object_df.select_dtypes(include=['object'])

# 对这198列试着转成时间类型,
raw_date_obj_cols_df = raw_objs_others_cols_df.apply(pd.to_datetime, errors='ignore')
# (30000, 105)     ---------------> 时间类型列
raw_date_cols_df = raw_date_obj_cols_df.select_dtypes(include=['datetime64'])
# (30000, 93)      ---------------> 字符类型列
raw_objs_cols_df = raw_date_obj_cols_df.select_dtypes(include=['object'])

##################################    关联字典表,进行特征初步筛选    ##############################################

# 将这198列数据存储下来,打开看看内容,进行特征初步筛选
# raw_objs_cols_df.to_csv('/Users/hushiwei/Downloads/object_data.csv', index=False)

# 将字典文件读取进来,关联上刚刚那198列object类型,看看都是些什么内容
# dic_df=pd.read_csv('/Users/hushiwei/GEO/数据集/建模数据集/数据字典.csv').iloc[:,:2]
# 修改一下类名称,英文名,中文解释
# dic_df.columns=['en_name','zh_name']

# 字典表中没有gid,
# pd.DataFrame(raw_objs_others_cols_df.drop(['gid','apply_time'],axis=1).columns).head()
# obj_dic_df=pd.DataFrame(raw_objs_others_cols_df.columns)
# obj_dic_df.columns=['en_name']

# 将这两份数据进行关联,查看object类型都是些什么数据
# 存储下来,查看对吧,进行初步筛选
# tmp_dic_dic_df = pd.merge(dic_df,obj_dic_df,on='en_name')
# tmp_dic_dic_df.to_csv('/Users/hushiwei/Downloads/object_data_dic.csv', index=False)

###################################################################################################################
#######################################     特征初筛,做衍生变量     ##################################################

# 从申请时间列做衍生变量
# raw_date_cols_df['apply_time'].dt.month

# 0           gid
# 1    apply_time

seed = 7
test_size = 0.40
# X_train, X_test, y_train, y_test = train_test_split(raw_numeric_cols_df, Y, test_size=test_size, random_state=seed)




# 取top10
# df.nlargest(10,'fscore')
# importance_numeric_columns=df.nlargest(50,'fscore').feature.values.tolist()
# new_numeric_dataset=raw_numeric_cols_df[importance_numeric_columns]
# 调试过程,将new_numeric_dataset存下来,免得又跑一遍才能得到这个结果
# new_numeric_dataset.to_pickle('./datas/new_numeric_dataset.pkl')
new_numeric_dataset=pd.read_pickle('./datas/new_numeric_dataset.pkl')
# X_train, X_test, y_train, y_test = train_test_split(new_numeric_dataset, Y, test_size=test_size, random_state=seed)

######################################  衍生变量  ###############################################

# 现在还剩时间类型和字符串类型的特征
print("#"*100)

apply_time_df=raw_date_cols_df['apply_time']


# 单日最大查询次数日期 1,3,6,12月
# 单日最大查询公司数日期 1,3,6,12月
# 单日最大查询次数日期 1,2,3,4,5,6,7,8,9,10...24周
# 单日最大查询公司数日期 1,2,3,4,5,6,7,8,9,10...24周
# 查询最早日期 1...24周
# 查询最近日期 1...24周
'''
时间衍生规则:
将查询最近日期的每一列与申请日期相减,然后做分箱操作.
依次类推将其余几个维度的日期也这样子转了
'''
# 从查询最早日期开始做起
# 列名以GDIW0015_W1-24开头
GDIW0015_cols=[col for col in raw_date_cols_df if col.startswith('GDIW0015_')]
# 取出查询最早日期的24列数据
GDIW0015_df=raw_date_cols_df[GDIW0015_cols]
# 将这24列数据与申请日期相减
GDIW0015_finall=GDIW0015_df.subtract(apply_time_df,axis=0)
# 通过后向填充的方式,填充空日期的位置,然后将时间转成float类型
GDIW0015_fillNa=GDIW0015_finall.bfill().astype('timedelta64[D]')
# 接着对齐进行分箱,这样就对日期转成分箱列特征了
filnal_GDIW015_df=GDIW0015_fillNa.apply(lambda x:pd.qcut(x,7,duplicates='drop',labels=False),axis=0)


def turn_date_to_bin(row_df,cols_prefix,apply_time_df):
    date_df_columns = [col for col in row_df if col.startswith(cols_prefix)]
    # 取出查询最早日期的24列数据
    date_df = row_df[date_df_columns]
    # 将这24列数据与申请日期相减
    date_df_have_NaT = date_df.subtract(apply_time_df, axis=0)
    # 通过后向填充的方式,填充空日期的位置,然后将时间转成float类型
    date_df_no_NaT = date_df_have_NaT.bfill().astype('timedelta64[D]')
    # 接着对齐进行分箱,这样就对日期转成分箱列特征了
    finaly_date_df = date_df_no_NaT.apply(lambda x: pd.qcut(x, 7, duplicates='drop', labels=False), axis=0)

    return finaly_date_df

# 查询最近日期 1...24周
# GDIW0016_W
filnal_GDIW0016_df = turn_date_to_bin(raw_date_cols_df,'GDIW0016_W',apply_time_df)


# 最早查询公司名称 1...24周
# 最近查询公司名称 1...24周


'''
字符类,主要是公司名称
'''




######################################################################################################################

# 将之前抽取的50个最重要的数值列与刚刚分箱的列拼接,构造成新的数据集进行模型训练
X=pd.concat([new_numeric_dataset,filnal_GDIW015_df,filnal_GDIW0016_df],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


######################################      XGBoost抽取Top200         ##############################################

# 将数值类型的数据入模,挑选出重要性比较高的特征top200
xgb = XGBClassifier(max_depth=5,max_features='sqrt', subsample=0.8, random_state=10)
xgb.fit(X_train, y_train)
#
print("train score : ", xgb.score(X_train, y_train))
print("test score : ", xgb.score(X_test, y_test))
print("train ks : ", ks_statistic(y_train, xgb.predict_proba(X_train)[:, 1]))
print("test ks : ", ks_statistic(y_test, xgb.predict_proba(X_test)[:, 1]))
#
# 根据xgboost抽取出特征重要性的top200
importance = xgb.get_booster().get_score()
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=False)
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])













########################################  最后进行交叉验证,开始调参  #########################################
# tuning parameters
# from sklearn.model_selection import GridSearchCV
# parameters = [{'n_estimators': [10, 100]},
#               {'learning_rate': [0.1, 0.01, 0.5]}]
# grid_search = GridSearchCV(estimator = gbm, param_grid = parameters, scoring='accuracy', cv = 3, n_jobs=-1)
# grid_search = grid_search.fit(train_X, train_y)
#
# gbm=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bytree=1, gamma=0, learning_rate=0.5, max_delta_step=0,
#        max_depth=3, max_features='sqrt', min_child_weight=1, missing=None,
#        n_estimators=100, n_jobs=1, nthread=None,
#        objective='binary:logistic', random_state=10, reg_alpha=0,
#        reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
#        subsample=0.8).fit(train_X, train_y)
# y_pred = gbm.predict(X_test_encode.as_matrix())
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
