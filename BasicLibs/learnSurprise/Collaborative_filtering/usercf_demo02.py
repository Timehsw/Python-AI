# -- encoding:utf-8 --
"""
Create by ibf on 2018/6/29
"""

from surprise import Dataset
from surprise import KNNBaseline
from surprise import evaluate

# 1. 数据加载(按行加载数据，一行一条用户对于物品的平方信息)
# a. 方式一：直接通过surprise框架加载默认的MovieLens数据
"""
load_builtin: 默认会从网络下载电影的评分数据，默认会保存到"~/.surprise_data"文件夹中
name参数可选值：'ml-100k', 'ml-1m', and 'jester'
"""
data = Dataset.load_builtin(name='ml-100k')

# 2. 做一个数据的交叉划分
data.split(5)

# 3. 模型构建
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
algo = KNNBaseline(k=20, min_k=1, sim_options=sim_options, bsl_options=bsl_options)


# 4. 模型效果评估
evaluate(algo, data=data, measures=['RMSE', 'MAE', 'FCP'])
