# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019/6/12
    Desc : 
    Note : 
'''

import os
from surprise import Dataset, Reader
from surprise import KNNBasic, KNNWithMeans, KNNBaseline

# data = Dataset.load_builtin(name='ml-100k')
file_path='/mnt/d/PycharmProjects/Python-AI/Recommend/datas/u.data'
reader=Reader(line_format='user item rating timestamp',sep='\t')
data=Dataset.load_from_file(file_path=file_path,reader=reader)
trainset = data.build_full_trainset()

sim_options = {
    'name': 'pearson',
    'user_based': True
}

bsl_options={
    'method':'sgd',
    'n_epochs':50,
    'reg':0.02,
    'learning_rate':0.01
}

algo=KNNBaseline(k=2,min_k=1,sim_options=sim_options,bsl_options=bsl_options)

# 模型训练
algo.fit(trainset)

# 模型预测
uid='196'
iid='242'
pred=algo.predict(uid,iid,4)
print("评分:{}".format(pred))
print("评分:{}".format(pred.est))

# same as below
# algo.estimate(algo.trainset.to_inner_uid(uid),algo.trainset.to_inner_iid(iid))