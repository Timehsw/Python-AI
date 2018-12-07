# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/12/7
    Desc : 
    Note : 
'''

import h2o

h2o.init()

train_path = './datas/higgs_train_imbalance_100k.csv'
test_path = './datas/higgs_test_imbalance_100k.csv'
df_train = h2o.import_file(train_path)
df_test = h2o.import_file(test_path)
# Transform first feature into categorical feature
df_train[0] = df_train[0].asfactor()
df_test[0] = df_test[0].asfactor()

param = {
    "ntrees": 100
    , "max_depth": 10
    , "learn_rate": 0.02
    , "sample_rate": 0.7
    , "col_sample_rate_per_tree": 0.9
    , "min_rows": 5
    , "seed": 4241
    , "score_tree_interval": 100
}
from h2o.estimators import H2OXGBoostEstimator

model = H2OXGBoostEstimator(**param)
model.train(x=list(range(1, df_train.shape[1])), y=0, training_frame=df_train, validation_frame=df_test)

prediction = model.predict(df_test)[:,2]
print(prediction)