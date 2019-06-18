# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019/6/12
    Desc :
    Note :
'''

import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.fpm import FPGrowth
from pyspark.sql import SQLContext, Row



# 1. 创建上下文
conf = SparkConf() \
    .setMaster('local[*]') \
    .setAppName('fp tree')
sc = SparkContext(conf=conf)
sql_context = SQLContext(sparkContext=sc)

# 2. 构建RDD（即交易数据组成的RDD对象，RDD中一条数据就是一条交易，是一个数组的形式）
items = [
    ['A', 'B', 'C', 'E', 'F', 'O'],
    ['A', 'C', 'G'],
    ['E', 'I'],
    ['A', 'C', 'D', 'E', 'G'],
    ['A', 'C', 'E', 'G', 'L'],
    ['E', 'J'],
    ['A', 'B', 'C', 'E', 'F', 'P'],
    ['A', 'C', 'D'],
    ['A', 'C', 'E', 'G', 'M'],
    ['A', 'C', 'E', 'G', 'N']
]
rdd = sc.parallelize(items)

# 3. 模型构建
# minSupport: 最小支持度
model = FPGrowth.train(rdd, minSupport=0.2, numPartitions=10)

# 4. 获取所有频繁项组成的RDD
all_freq_itemsets_rdd = model.freqItemsets()
all_freq_itemsets_rdd.cache()
print(all_freq_itemsets_rdd.collect())

# 5. 获取长度为3的频繁项集组成的RDD
three_freq_itemsets_rdd = all_freq_itemsets_rdd \
    .filter(lambda freq_itemset: len(freq_itemset.items) == 3)
print(three_freq_itemsets_rdd.collect())

# 6. 获取存在A的长度为3的最大的频繁项集
max_freq_of_include_a_three_itemsets = three_freq_itemsets_rdd \
    .filter(lambda freq_itemset: 'A' in freq_itemset.items) \
    .max(key=lambda freq_itemset: freq_itemset.freq)
print(max_freq_of_include_a_three_itemsets)

# 7. 将结果保存数据库(表名称: tb_fp_tree_result, 结构: lhs, rhs, support)
all_freq_itemsets_row_rdd = all_freq_itemsets_rdd \
    .filter(lambda freq_itemset: len(freq_itemset.items) == 3) \
    .flatMap(
    lambda freq_itemset: map(lambda item: Row(lhs=item, rhs=','.join([i for i in freq_itemset.items if i != item]),
                                              support=freq_itemset.freq), freq_itemset.items))
all_freq_itemsets_df = sql_context.createDataFrame(all_freq_itemsets_row_rdd)
all_freq_itemsets_df.show(truncate=False)
