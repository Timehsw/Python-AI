# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/25
    Desc : 
    Note : 
'''

from pyspark.sql import SparkSession
import pandas as pd
import time

def _map_to_pandas(rdds):
    """ Needs to be here due to pickling issues """
    return [pd.DataFrame(list(rdds))]


def toPandas(df, n_partitions=None):
    """
    Returns the contents of `df` as a local `pandas.DataFrame` in a speedy fashion. The DataFrame is
    repartitioned if `n_partitions` is passed.
    :param df:              pyspark.sql.DataFrame
    :param n_partitions:    int or None
    :return:                pandas.DataFrame
    """
    if n_partitions is not None: df = df.repartition(n_partitions)
    df_pand = df.rdd.mapPartitions(_map_to_pandas).collect()
    df_pand = pd.concat(df_pand)
    df_pand.columns = df.columns
    return df_pand


if __name__ == "__main__":
    # $example on:init_session$
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    # $example off:init_session$

    spark_df=spark.read.csv('/Users/hushiwei/Downloads/bigdata.csv')
    spark_df.show()
    start=time.time()

    print('`````````````````````````````````')
    # df=spark_df.toPandas()
    df=toPandas(spark_df)
    print(df.head())
    print('cost ',time.time()-start)