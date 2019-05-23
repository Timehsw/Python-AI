# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-05-06
    Desc : https://blog.csdn.net/xyf123/article/details/78088042
    Note : http://dl.bintray.com/spark-packages/maven/graphframes/graphframes/
'''
import os
import traceback
from time import localtime, strftime

os.environ['SPARK_HOME'] = '/Users/hushiwei/devApps/spark-2.3.1-bin-hadoop2.7'
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell'
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import udf
from BasicLibs.learnPySpark.graphframes.MysqlUtils import MySQL


def insert_table(content_dic):
    try:
        handler = MySQL(HOST, USERNAME, PASSWORD, PORT)
        handler.selectDb(DB)
        table_name = "community_detection"

        handler.insert(table_name, content_dic)
        handler.commit()
        return True
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return False


sc = pyspark.SparkContext("local[*]")
spark = SparkSession.builder.appName('notebook').getOrCreate()

from graphframes import *
import hashlib

# Load sample webgraph
raw_data = spark.read.csv(
    path="file:///Users/hushiwei/PycharmProjects/Python-AI/BasicLibs/learnPySpark/graphframes/data/res.txt", sep="\t",
    header=True)
print(raw_data.count())

# Select set of parents and children TLDs (your nodes) to assign id for each node.

# aggcodes = raw_data.select("parentTLD", "childTLD").rdd.flatMap(lambda x: x).distinct()
aggcodes = raw_data.select("start_id", "end_id").rdd.flatMap(lambda x: x).distinct()
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

from pyspark.sql.functions import desc

# 获取最多的label top1
communities.groupBy("label").count().sort(desc('count')).show()
# max_label,count=communities.groupBy("label").count().sort(desc('count')).take(1)[0]
# 暂定当社区里面大于7个人就是一个社区了,过滤出来
tmp_communities = communities.groupBy("label").count().sort(desc('count')).filter("count>7")
# 查出这些社区的人来
labels_counts = [row.asDict() for row in tmp_communities.rdd.collect()]
# communities.filter("label==77309411379").select("id").show()

print(f"There are {communities.select('label').distinct().count()} communities in sample graph.")

res = {}
for one in labels_counts:
    label = one['label']
    num = one['count']
    ids = [row['name'] for row in communities.filter("label==%s" % (label)).select("name").collect()]
    res[str(label)] = ids

    print("Community {} has {} nums. which has people {}".format(label, num, ','.join(ids)))

HOST = "10.111.32.118"
USERNAME = "klg"
PASSWORD = "klgkei9d&"
PORT = 3306
DB = "klg"

final_result = {}
final_result["model_id"] = 8888
final_result["result"] = res
final_result["create_date"] = strftime("%Y-%m-%d %H:%M:%S", localtime())

insert_table(final_result)
