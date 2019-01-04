# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/9/19
    Desc : 
    Note : 
'''

from pyspark.sql import SparkSession
# $example off:init_session$

# $example on:schema_inferring$
from pyspark.sql import Row
# $example off:schema_inferring$

# $example on:programmatic_schema$
# Import data types
from pyspark.sql.types import *


# $example off:programmatic_schema$


def lb(lb):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(lb)
    return [i for i in le.transform(lb)]


def schema_inference_example(spark):

    # Load a text file and convert each line to a Row.
    df = spark.read.csv("./resources/userab.csv", inferSchema=True, header=True)

    df.rdd.map(lambda x: [(k, v) for k, v in x.asDict().items()]).map(lambda x: [i for i in x]).flatMap(
        lambda x: x).groupBy(lambda x: x[0]).map(lambda x: (lb([i[1] for i in x[1]]))).foreach(
        lambda x: print(x, '==='))



if __name__ == "__main__":
    # $example on:init_session$
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    # $example off:init_session$

    # basic_df_example(spark)
    schema_inference_example(spark)
    # programmatic_schema_example(spark)

    spark.stop()
