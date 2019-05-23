# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : 导入某一列数据进入Collection中
    Note : ContactAddrs
'''

import os
import pandas as pd
import sys
import json

from arango import ArangoClient

root_path = "/Users/hushiwei/Downloads/模拟数据"
os.chdir(root_path)


def create_save_collections(featureColleciton, collection_name, df, is_turn_to_id=True):
    '''
    将df中的feature导入到Collections中
    如果内容是中文则将中文装成id,并将这字典表存储起来.因为arangodb中不能将中文作为_key

    :param featureColleciton:
    :param collection_name:
    :param df:
    :param is_turn_to_id:若为true,则将特征的编码id做为_key.否则直接将特征内容作为_key
    :return:
    '''
    feature_df = df[collection_name]
    feature_df.drop_duplicates(inplace=True)
    if is_turn_to_id:
        result = [{'_key': str(index), 'name': value} for index, value in feature_df.iteritems()]
        feature_id = {work_name: str(index) for index, work_name in feature_df.iteritems()}
        with open("./%s_id.json" % (collection_name), "w") as fp:
            json.dump(feature_id, fp)
    else:
        result = [{'_key': str(value), 'name': value} for value in feature_df]

    featureColleciton.insert_many(result)


def get_collection_hander(db, collection_name):
    '''
    在db数据库下创建获取collection
    :param db:
    :param collection_name:
    :return:
    '''
    if db.has_collection(collection_name):
        feature_colleciton = db.collection(collection_name)
    else:
        feature_colleciton = db.create_collection(collection_name)
    return feature_colleciton


def get_df(path):
    '''
    读取数据
    :param path:
    :return:
    '''
    data_df = pd.read_csv(path, error_bad_lines=False)
    return data_df


clinet = ArangoClient(host="10.111.32.65")
# db = clinet.db('financerisk', username="root", password="hxixpLi")
db = clinet.db('financerisk', username="hushiwei", password="hushiwei")

data_df = get_df("apply_nomissing.csv")

# 中文处理转id的
# collections_names = ['WorkName','WorkAddr','ContactAddr','HomeAddr', 'IdAddr', 'HouseAddr']
# feature_names = ['workName','workAddr','contactAddr','homeAddr', 'idAddr', 'houseAddr']
# for i in range(6):
#     print("~" * 10, "start create %s collections" % (collections_names[i]), "~" * 10)
#     collection_hander = get_collection_hander(db, collections_names[i])
#     create_save_collections(collection_hander, feature_names[i], data_df)

#####################################################################################################

# 直接入库的
collections_names = ['CustomNum', 'Phone', 'MobilePhone', 'RelationCustomNum', 'RelationMobile']
feature_names = ['customNum', 'phone', 'mobilePhone', 'relationCustomNum', 'relationMobile']

for i in range(4):
    print("~" * 10, "start create %s collections" % (collections_names[i]), "~" * 10)
    collection_hander = get_collection_hander(db, collections_names[i])
    create_save_collections(collection_hander, feature_names[i], data_df, is_turn_to_id=False)
