# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019/6/18
    Desc : 
    Note : 
'''
from surprise import Dataset, Reader
from surprise import KNNBaseline
import pickle

# 1. 读取数据

file_path='/mnt/d/PycharmProjects/Python-AI/Recommend/datas/u.data'

reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path=file_path, reader=reader)

# 2. 数据基本的处理
trainset = data.build_full_trainset()

# 3. 模型构建
# 使用sgd梯度下降的方式计算
bsl_options = {
    'method': 'sgd'
}
# 给定相似度的计算参数
sim_options = {
    'name': 'pearson',
    'user_based': False
}
algo = KNNBaseline(k=40, min_k=1, sim_options=sim_options, bsl_options=bsl_options)

# 4. 模型训练
algo.fit(trainset)

# 5. 保存模型训练出来的物品与物品之间的相似度信息，这里只针对于每个物品保证最相似度的10个其它物品以及对应的相似度
item_2_items = {}
# 获取总的物品数目
total_items = algo.trainset.n_items
print("总物品数目为:{}".format(total_items))
k = 10
for inner_item_id1 in range(total_items):
    # 将内部id转换为外部id
    raw_item_id1 = algo.trainset.to_raw_iid(inner_item_id1)
    # 构建一个针对于当前物品id的相似列表
    result_list = []
    for inner_item_id2 in range(total_items):
        if inner_item_id1 != inner_item_id2:
            # 1. 获取相似度
            sim = algo.sim[inner_item_id1][inner_item_id2]
            # 2. 将结果添加集合
            result_list.append((sim, algo.trainset.to_raw_iid(inner_item_id2)))
    # 从集合中获取相似度最大的K个数据
    result_list.sort(key=lambda t: t[0])
    result_list = result_list[-k:]
    # 结果保存字典
    item_2_items[raw_item_id1] = result_list
# 最终结果保存
# with open('./result/item_2_item_sim.pkl', 'wb') as writer:
#     pickle.dump(item_2_items, writer)
print("Done!!!!")