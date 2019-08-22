# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-08-02
    Desc : 特征重要性的5种参数
    Note :
'''
import numpy as np

sample_num = 10
feature_num = 2

np.random.seed(0)
data = np.random.randn(sample_num, feature_num)
np.random.seed(0)
label = np.random.randint(0, 2, sample_num)

import xgboost as xgb

train_data = xgb.DMatrix(data, label=label)
params = {'max_depth': 3}
bst = xgb.train(params, train_data, num_boost_round=1)

for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))


xgb.to_graphviz(bst, num_trees=0)
