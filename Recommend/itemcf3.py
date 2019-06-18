# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019/6/12
    Desc : 
    Note : 
'''

import surprise
from surprise import Dataset, Reader
from surprise import KNNBasic, KNNWithMeans, KNNBaseline

dump_model_to_file = False

# 1. 读取数据
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path='../datas/u.data', reader=reader)

# 2. 数据基本的处理
trainset = data.build_full_trainset()

# 3. 模型构建
# 使用sgd梯度下降的方式计算
bsl_options = {
    'method': 'sgd',
    'n_epochs': 10,  # 迭代次数
    'reg': 0.02,  # 正则化项系数，一般比较小
    'learning_rate': 0.1  # 梯度下降中的学习率
}
# 给定相似度的计算参数
sim_options = {
    'name': 'pearson',  # 指定计算相似度的方式，可选值为: cosine、msd、pearson、pearson_baseline
    'user_based': False  # 设置为True表示使用基于用户的协同过滤算法，设置为False表示基于物品的协同过滤算法
}
algo = KNNBaseline(k=40, min_k=1, sim_options=sim_options, bsl_options=bsl_options)

# 4. 模型训练
algo.fit(trainset)

# 5. 模型持久化
# TODO: 最终输出的时候，不输出所有的评分结果，只输出对于当前用户而言，评分最高的30个商品以及对应的评分信息
# TODO: 如果将推荐结果保存到数据库中，数据库选择MySQL
if dump_model_to_file:
    # 直接模型持久化为磁盘文件(要求输出的磁盘文件必须是存在的)
    surprise.dump.dump(file_name='./result/itemcf03.m', algo=algo)
else:
    # 直接将推荐结果持久化, 这里直接输出所有用户对于所有物品的评分
    file_path = './result/user_item_rating2.txt'
    with open(file_path, 'w') as writer:
        total_users = algo.trainset.n_users
        total_items = algo.trainset.n_items
        print("总用户数目:{}, 总物品数目:{}".format(total_users, total_items))
        for inner_user_id in range(total_users):
            # 将内部用户id转换为外部用户id
            raw_user_id = algo.trainset.to_raw_uid(inner_user_id)
            # 获取当前用户raw_user_id所评估的所有商品id所组成的集合
            all_user_inner_item_ids = [iid for (iid, _) in algo.trainset.ur[inner_user_id]]
            for inner_item_id in range(total_items):
                # 将内部商品id转换为外部商品id
                raw_item_id = algo.trainset.to_raw_iid(inner_item_id)
                # 做一个数据过滤，要求raw_item_id是用户raw_user_id没有评论过的
                if inner_item_id not in all_user_inner_item_ids:
                    # 做一个预测评分
                    rating = algo.predict(raw_user_id, raw_item_id).est
                    # 结果输出
                    writer.writelines('%s\t%s\t%.3f\n' % (raw_user_id, raw_item_id, rating))
                # # 做一个预测评分
                # rating = algo.predict(raw_user_id, raw_item_id).est
                # # 结果输出
                # writer.writelines('%s\t%s\t%.3f\n' % (raw_user_id, raw_item_id, rating))

print("Done!!!")