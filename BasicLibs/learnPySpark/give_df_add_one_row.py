# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/9/20
    Desc : 
    Note : 
'''

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *


def basic_df_example(spark):
    df = spark.read.json("./resources/people.json")
    df.show()

    df1=spark.createDataFrame([Row(age=25,name='hushiwei'),Row(age=60,name='Jack')])
    df1.show()

    df2=df.union(df1)
    df2.show()


if __name__ == "__main__":
    # $example on:init_session$
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    basic_df_example(spark)

    spark.stop()
