# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-01-21
    Desc : 
    Note : 
'''

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, MapType, ArrayType, StructType
from pyspark2pmml import PMMLBuilder
from pyspark.ml.classification import RandomForestClassifier

if __name__ == '__main__':
    spark = SparkSession.builder \
        .config("spark.jars", "/Users/hushiwei/PycharmProjects/Python-AI/BasicLibs/xgboost4j/xgboost4j-0.81.jar")\
        .config("spark.driver-class-path","/Users/hushiwei/PycharmProjects/Python-AI/BasicLibs/xgboost4j/xgboost4j-0.81.jar")\
        .config("spark.py-files", "/Users/hushiwei/PycharmProjects/Python-AI/BasicLibs/xgboost4j/xgboost4j-0.81.jar")\
        .master("local[*]").appName("xgboost4j").getOrCreate()

    sc = spark.sparkContext

    xgboost = sc._jvm.ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier()
    print(xgboost)