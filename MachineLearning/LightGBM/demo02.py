# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-02-13
    Desc : 
    Note : 
'''

import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import sys

try:
    import cPickle as pickle
except BaseException:
    import pickle

print('Loading data...')
# load or create your dataset
df_train = pd.read_csv('./datas/binary.train', header=None, sep='\t')
df_test = pd.read_csv('./datas/binary.test', header=None, sep='\t')
W_train = pd.read_csv('./datas/binary.train.weight', header=None)[0]
W_test = pd.read_csv('./datas/binary.test.weight', header=None)[0]

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

num_train, num_feature = X_train.shape

# create dataset for lightgbm
# if you want to re-use data, remember to set free_raw_data=False
lgb_train = lgb.Dataset(X_train, y_train,
                        weight=W_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,
                       weight=W_test, free_raw_data=False)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# generate feature names
feature_name = ['feature_' + str(col) for col in range(num_feature)]

print('Starting training...')
# feature_name and categorical_feature
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                valid_sets=lgb_train,  # eval training data
                feature_name=feature_name,
                categorical_feature=[21])

print('Finished first 10 rounds...')
# check feature name
print('7th feature name is:', lgb_train.feature_name[6])

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Dumping model to JSON...')
# dump model to JSON (and save to file)
model_json = gbm.dump_model()

with open('model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)


def parseOneTree(root, index, array_type='double', return_type='double'):
    def ifElse(node):
        if 'leaf_index' in node:
            return 'return ' + str(node['leaf_value']) + ';'
        else:
            condition = 'arr[' + str(node['split_feature']) + ']'
            if node['decision_type'] == '<=':
                condition += ' <= ' + str(node['threshold'])
            else:
                condition += ' == ' + str(node['threshold'])
            left = ifElse(node['left_child'])
            right = ifElse(node['right_child'])
            return 'if ( ' + condition + ' ) { ' + left + ' } else { ' + right + ' }'

    return return_type + ' predictTree' + str(index) + '(' + array_type + '[] arr) { ' + ifElse(root) + ' }'


def parseAllTrees(trees, array_type='double', return_type='double'):
    return '\n\n'.join(
        [parseOneTree(tree['tree_structure'], idx, array_type, return_type) for idx, tree in enumerate(trees)]) \
           + '\n\n' + return_type + ' predict(' + array_type + '[] arr) { ' \
           + 'return ' + ' + '.join(['predictTree' + str(i) + '(arr)' for i in range(len(trees))]) + ';' \
           + '}'


with open('./model/if.else', 'w+') as f:
    f.write(parseAllTrees(model_json["tree_info"]))

# 进行预测

test_point=gbm.predict(np.array([[0.289, 0.823, 0.013, 0.615, -1.601, 0.177, 2.403, -0.015, 0.000, 0.258, 1.151, 1.036, 2.215, 0.694, 0.553, -1.326, 2.548, 0.411, 0.366, 0.106, 0.000, 0.482, 0.562, 0.989, 0.670, 0.404, 0.516, 0.561],[0.455,-0.880	,-1.482	,1.260,	-0.178	,1.499	,0.158,	1.022,	0.000,	1.867,	-0.435	,-0.675	,2.215,	1.234,	0.783,	1.586,	0.000,	0.641	,-0.454,	-0.409,	3.102,	1.002	,0.964,	0.986,	0.761,	0.240,	1.190,	0.995]]),num_iteration=gbm.best_iteration)

print(test_point)

test_point=gbm.predict(np.array([[0.455,-0.880	,-1.482	,1.260,	-0.178	,1.499	,0.158,	1.022,	0.000,	1.867,	-0.435	,-0.675	,2.215,	1.234,	0.783,	1.586,	0.000,	0.641	,-0.454,	-0.409,	3.102,	1.002	,0.964,	0.986,	0.761,	0.240,	1.190,	0.995]]),num_iteration=gbm.best_iteration)

print(test_point)