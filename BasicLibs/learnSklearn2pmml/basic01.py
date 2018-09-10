# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/9/10
    Desc : sklearn2pmml案例
    Note : 
'''
import numpy as np
import pandas
from sklearn2pmml import PMMLPipeline,sklearn2pmml
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression

heart_data = pandas.read_csv("heart.csv")
# 用Mapper定义特征工程
mapper = DataFrameMapper([
    (['sbp'], MinMaxScaler()),
    (['tobacco'], MinMaxScaler()),
    ('ldl', None),
    ('adiposity', None),
    (['famhist'], LabelBinarizer()),
    ('typea', None),
    ('obesity', None),
    ('alcohol', None),
    (['age'], FunctionTransformer(np.log)),
])

# 用pipeline定义使用的模型，特征工程等
pipeline = PMMLPipeline([
    ('mapper', mapper),
    ("classifier", LinearRegression())
])

pipeline.fit(heart_data[heart_data.columns.difference(["chd"])], heart_data["chd"])
# 导出模型文件
sklearn2pmml(pipeline, "lrHeart.xml", with_repr=True)
