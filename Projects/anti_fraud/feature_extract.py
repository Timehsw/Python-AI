# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/3
    Desc : 特征抽取
'''

import numpy as np
import pandas as pd
import sys

df = pd.read_csv('./datas/LoanStats3a.csv', skiprows=1, low_memory=True)

# print(df.head())
# print(df.shape)

# print(df.info())
# 按列查看空值分布
# print(df.isnull().sum())

# 删除肉眼可见的空值列
df.drop(['id', 'member_id'], axis=1, inplace=True)

# 通过开启replace的正则功能,替换匹配到的地方
df['term'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
df['int_rate'].replace('%', value='', inplace=True)

# 两列评级,删除一列,留一列即可
df.drop('sub_grade', axis=1, inplace=True)

df.drop('emp_title', axis=1, inplace=True)

df.emp_length.replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)

# 删除行列都是空的数据
df.dropna(axis=1, how='all', inplace=True)
df.dropna(axis=0, how='all', inplace=True)

df.drop(['debt_settlement_flag_date', 'settlement_status', 'settlement_date', 'settlement_amount',
         'settlement_percentage', 'settlement_term'], axis=1, inplace=True)

# 删除float类型中重复值较多的特征
# select_dtypes 用于选择类型
for col in df.select_dtypes(include=['float']).columns:
    print('col: {} has {}'.format(col, len(df[col].unique())))

df.drop(['delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec',
         'total_acc', 'out_prncp', 'out_prncp_inv', 'collections_12_mths_ex_med', 'policy_code', 'acc_now_delinq',
         'chargeoff_within_12_mths', 'delinq_amnt', 'pub_rec_bankruptcies', 'tax_liens'], axis=1, inplace=True)

print('-' * 60)
# 删除object类型中重复值较多的特征
# select_dtypes 用于选择类型
for col in df.select_dtypes(include=['float']).columns:
    print('col: {} has {}'.format(col, len(df[col].unique())))

print('-' * 60)

for col in df.select_dtypes(include=['object']).columns:
    print('col: {} has {}'.format(col, len(df[col].unique())))

df.drop(['term', 'grade', 'emp_length', 'home_ownership', 'verification_status', 'issue_d', 'pymnt_plan', 'purpose',
         'zip_code', 'addr_state', 'earliest_cr_line', 'initial_list_status', 'last_pymnt_d', 'next_pymnt_d',
         'last_credit_pull_d', 'application_type', 'hardship_flag', 'disbursement_method', 'debt_settlement_flag'],
        axis=1, inplace=True)
print('-' * 60)

# 最后再整体筛选一次
df.drop(['desc', 'title'], axis=1, inplace=True)

# 对标签Y值进行处理
print(df.loan_status.value_counts())
df.loan_status.replace('Fully Paid', int(1), inplace=True)
df.loan_status.replace('Charged Off', int(0), inplace=True)
df.loan_status.replace('Does not meet the credit policy. Status:Fully Paid', np.nan, inplace=True)
df.loan_status.replace('Does not meet the credit policy. Status:Charged Off', np.nan, inplace=True)

# 删掉loan_status列中是np.nan的所在行
# 也就是删除标签值为空值的实例
df.dropna(subset=['loan_status'], axis=0, how='any', inplace=True)

# 用0去填充所有的空值
df.fillna(0, inplace=True)
df.fillna(0.0, inplace=True)
# print(df.loan_status.value_counts())

# print(df.head())
# print(df.info())

print('-' * 20, '以上为数据清洗部分', '-' * 20)

# 检测清洗后的样本特征的相关性,去除多重相关性特征(保留1列)
# 接着清除数据相关的列

# 相关系数
cor = df.corr()

# tril 返回矩阵的下三角
cor.iloc[:, :] = np.tril(cor, k=-1)
cor = cor.stack()
print(cor[(cor > 0.55) | (cor < -0.55)])

# 删除大于0.95以上的
df.drop(['loan_amnt', 'funded_amnt', 'total_pymnt'], axis=1, inplace=True)
print(df.info())

# 再次打印信息,查看是否有非float类型的数据,将其做哑变量处理
df = pd.get_dummies(df)
df.to_csv('./datas/feature_005.csv', index=None)
print('-' * 100)
# print(df.head(10))

print(df.info())
