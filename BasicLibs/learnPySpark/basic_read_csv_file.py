# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/16
    Desc : 
    Note : 
'''
from pyspark.sql import SparkSession

if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    path = './hh/'
    df = spark.read.csv(path)
    df.show(n=100,truncate=False)


