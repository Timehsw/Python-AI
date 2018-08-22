# -- encoding:utf-8 --
"""
Create by ibf on 2018/6/29
"""

import os
from surprise import Dataset, Reader
from surprise import KNNBasic, KNNWithMeans, KNNBaseline

"""
KNNBasic: 基础的协同过滤算法，直接对评分做加权来获取预测评分
KNNWithMeans: 在KNNBasic的基础上，计算评分和均值之间差值来加权计算预测评分；表示模型中考虑了用户的独特性
KNNBaseline：在KNNWithMeans的基础上，将均值更换为baseline的值，表示模型中考虑了用户的独特性和物品的独特性
备注：一般使用比较多的是KNNBaseline这个API
"""

# 1. 数据加载(按行加载数据，一行一条用户对于物品的平方信息)
# a. 方式一：直接通过surprise框架加载默认的MovieLens数据
"""
load_builtin: 默认会从网络下载电影的评分数据，默认会保存到"~/.surprise_data"文件夹中
name参数可选值：'ml-100k', 'ml-1m', and 'jester'
"""
# data = Dataset.load_builtin(name='ml-100k')

# 2. 方式二：直接从文件中加载数据，要求文件内容至少包含用户id、物品id以及评分rating这三列的数据
file_path = os.path.expanduser('../datas/u.data')
# 指定一个文件内容的读取器，根据数据的格式给定
# line_format：给定数据中一行数据由那几个信息组成
# sep：分割符，当前指定为\t
# rating_scale: 给定文件中rating的取值范围，默认是(1,5)，表示最小为1，最大为5
# skip_lines: 给定跳过文件的开头几行，默认为0
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path=file_path, reader=reader)

# 2. 数据的处理
# 因为读取进来的数据是一行一行的用户物品评分数据，所以需要转换为用户-物品的评分矩阵
trainset = data.build_full_trainset()

# 3. 模型构建
# k: 指定在产生预测值的时候，最多获取多少个相似用户/物品
# min_k: 指定在产生预测值的时候，最少需要多少个相似用户/物品
# sim_options: 给定相似度的计算方式的参数
# bsl_options: 给定计算baseline的值的计算方式的相关API
sim_options = {
    'name': 'pearson',  # 指定采用什么方式的相似度计算公式，可选值： cosine、msd、pearson、pearson_baseline
    'user_based': True  # 指定模型是UserCF还是ItemCF，设置为True表示是UserCF模型
}
bsl_options = {
    'method': 'sgd',  # 指定采用何种模型求解方式，默认为als，可选sgd
    'n_epochs': 50,  # 迭代次数
    'reg': 0.02,  # 正则化系数值
    'learning_rate': 0.01  # 梯度下降中，参数更新的学习率
}
algo = KNNBaseline(k=2, min_k=1, sim_options=sim_options, bsl_options=bsl_options)

# 4. 模型训练
algo.fit(trainset)

# 5. 模型预测
"""
在Surprise框架中，获取预测评分的API必须使用predict，predict底层会调用estimate API：
两个API的区别：
predict：传入的是实际的用户id和物品id，predict会处理负的评分(转换过程，也就是数据读取的反过程)
estimate: 传入的是转换之后的用户id和物品id，estimate不会处理
"""
uid = "196"
iid = "242"
pred = algo.predict(uid, iid, 3)
print("评分:{}".format(pred))
print("评分:{}".format(pred.est))
