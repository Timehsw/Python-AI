# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018-12-28
    Desc : 
    Note : 
'''
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
import logging, sys

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from pysparkling import *
from h2o.grid.grid_search import H2OGridSearch


from h2o.estimators import H2OXGBoostEstimator
spark = SparkSession.builder.getOrCreate()


df = spark.read.csv(path='hdfs://10.111.32.12:8020/user/dp/file/data/liyiwen/282/1918/a2a66398-2da2-4c4b-b1c0-8153743d9c6c/add_feature_column', header=True, inferSchema=True)


conf = H2OConf(spark)


hc = H2OContext.getOrCreate(spark,conf=conf)
h2o_df = hc.as_h2o_frame(df,framename="df_h2o")

model_gbm = H2OGradientBoostingEstimator(ntrees=100,max_depth=6,learn_rate=0.1)

predictors = h2o_df.names[:]
ratios = [0.6,0.2]
frs = h2o_df.split_frame(ratios,seed=12345)
train = frs[0]
train.frame_id = "Train"
valid = frs[2]
valid.frame_id = "Validation"
test = frs[1]
test.frame_id = "Test"
model_gbm.train(x=predictors,y="GEO_Y",training_frame=train,validation_frame=valid)
print(model_gbm.varimp())