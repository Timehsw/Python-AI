# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-05-06
    Desc : https://blog.csdn.net/xyf123/article/details/78088042
    Note : http://dl.bintray.com/spark-packages/maven/graphframes/graphframes/
'''
import os

os.environ['SPARK_HOME'] = '/Users/hushiwei/devApps/spark-2.3.1-bin-hadoop2.7'
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell'
# graphframes-0.6.0-spark2.3-s_2.11.jar
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import udf

sc = pyspark.SparkContext("local[*]")
spark = SparkSession.builder.appName('notebook').getOrCreate()

from graphframes import *
import hashlib

# Load sample webgraph
raw_data = spark.read.csv(
    path="file:///Users/hushiwei/PycharmProjects/Python-AI/BasicLibs/learnPySpark/graphframes/data/data.txt", sep="\t",
    header=True)
print(raw_data.count())



# Select set of parents and children TLDs (your nodes) to assign id for each node.

# aggcodes = raw_data.select("parentTLD", "childTLD").rdd.flatMap(lambda x: x).distinct()
aggcodes=raw_data.select("start_id","end_id").rdd.flatMap(lambda x: x).distinct()
print(aggcodes.count())


def hashnode(x):
    return hashlib.sha1(x.encode("UTF-8")).hexdigest()[:8]


hashnode_udf = udf(hashnode)

vertices = aggcodes.map(lambda x: (hashnode(x), x)).toDF(["id", "name"])

print(vertices.show(5))

edges = raw_data.select("start_id", "end_id") \
    .withColumn("src", hashnode_udf("start_id")) \
    .withColumn("dst", hashnode_udf("end_id")) \
    .select("src", "dst")

print(edges.show(5))

# create GraphFrame
graph = GraphFrame(vertices, edges)

# Label Propagation Algorithm

# Run LPA
communities = graph.labelPropagation(maxIter=5)
communities.persist().show(10)

print(f"There are {communities.select('label').distinct().count()} communities in sample graph.")
