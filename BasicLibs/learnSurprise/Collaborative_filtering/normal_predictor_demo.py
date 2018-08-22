# -- encoding:utf-8 --
"""
Create by ibf on 2018/6/29
"""

import os
from surprise import Dataset, Reader
from surprise import NormalPredictor

# 1. 数据加载(按行加载数据，一行一条用户对于物品的平方信息)
# a. 方式一：直接通过surprise框架加载默认的MovieLens数据
"""
load_builtin: 默认会从网络下载电影的评分数据，默认会保存到"~/.surprise_data"文件夹中
name参数可选值：'ml-100k', 'ml-1m', and 'jester'
"""
data = Dataset.load_builtin(name='ml-100k')

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
algo = NormalPredictor()

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
