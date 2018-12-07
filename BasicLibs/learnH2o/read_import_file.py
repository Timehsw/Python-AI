# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/12/6
    Desc : 
    Note : 
'''

import h2o
from h2o.estimators.kmeans import H2OKMeansEstimator

h2o.init()
# 从本地读取数据
# iris = h2o.upload_file('./datas/iris.data')

# 从远程读取数据
iris = h2o.import_file(path="https://github.com/h2oai/h2o-3/raw/master/h2o-r/h2o-package/inst/extdata/iris_wheader.csv")
bank_df = h2o.import_file(path="hdfs://10.111.32.12:8020/user/dp/file/data/hushiwei/107/1382/09d84956-5f8d-4201-b9bf-f2a2c9487374/input_dataset")

iris.describe()
