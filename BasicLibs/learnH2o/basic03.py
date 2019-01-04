# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018-12-29
    Desc : 
    Note : 
'''
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
from pysparkling import *

conf = H2OConf(spark).set_external_cluster_mode().use_auto_cluster_start().set_h2o_driver_path(
    "/Users/hushiwei/devApps/h2o-sparking/h2odriver-sw2.3.18-hdp2.6-extended.jar") \
    .set_num_of_external_h2o_nodes(1).set_mapper_xmx("2G").set_yarn_queue("default")
hc = H2OContext.getOrCreate(spark, conf)
