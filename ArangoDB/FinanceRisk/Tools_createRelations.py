# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : 
    Note : 创建关系
'''

# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : 
    Note : 
'''

from pyArango.connection import *
import os
import pandas as pd
import sys

from arango import ArangoClient

root_path = "/Users/hushiwei/Downloads/模拟数据"
os.chdir(root_path)


def create_graph_tool(db, graph_name):
    if db.has_graph(graph_name):
        graph = db.graph(graph_name)
    else:
        graph = db.create_graph(graph_name)
    return graph


def delete_graph(db, graph_name):
    db.delete_graph(graph_name)


def create_edge_hander(graph, from_collectins_names, to_collections_names="CustomIds"):
    edge_hander = graph.create_edge_definition(
        edge_collection=to_collections_names + "_" + from_collectins_names,
        from_vertex_collections=[from_collectins_names],
        to_vertex_collections=[to_collections_names]
    )
    return edge_hander


def get_df(path):
    '''
    读取数据
    :param path:
    :return:
    '''
    data_df = pd.read_csv(path, error_bad_lines=False)
    return data_df


def get_custom_and_feature_relation_df(feature_name, relation_feature_name, data_df, custom_name="customNum"):
    '''
    获取用户id和要关联的特征
    :param custom_name:
    :param feature_name:
    :param data_df:
    :return:
    '''
    return data_df[[custom_name, feature_name, relation_feature_name]]


def get_from_and_to_df(from_name, to_name, data_df):
    '''
    获取from---> to的数据
    :param custom_name:
    :param feature_name:
    :param data_df:
    :return:
    '''
    return data_df[[from_name, to_name]]


def get_custom_and_feature_df(feature_name, data_df, custom_name="customNum"):
    '''
    获取用户id和要关联的特征
    :param custom_name:
    :param feature_name:
    :param data_df:
    :return:
    '''
    return data_df[[custom_name, feature_name]]


def get_dict_id(feature_name):
    with open("./%s_id.json" % (feature_name), "r") as fp:
        dic = json.load(fp)
    return dic


def create_relation_feature_and_feature(edge_hander, collection_name, feature_name, data_df, is_turn_to_id=True):
    '''
    两列特征之间建立关系
    :param edge_hander:
    :param collection_name:
    :param feature_name:
    :param data_df:
    :param is_turn_to_id:
    :return:
    '''
    result = []
    if is_turn_to_id:
        from_feature_id_dic = get_dict_id(feature_name[0])
        to_feature_id_dic = get_dict_id(feature_name[1])
        for index, row in data_df.iterrows():
            dic = {
                '_from': "%s/%s" % (collection_name[0], from_feature_id_dic[row[feature_name[0]]]),
                '_to': "%s/%s" % (collection_name[1], to_feature_id_dic[row[feature_name[1]]])
            }
            # result.append(dic)
            edge_hander.insert(dic)
    else:

        for index, row in data_df.iterrows():
            dic = {
                '_from': "%s/%s" % (collection_name[0], row[feature_name[0]]),
                '_to': "%s/%s" % (collection_name[1], row[feature_name[1]])
            }
            # result.append(dic)
            try:
                edge_hander.insert(dic)
            except:
                print(dic)


def create_relation_custom_and_feature(edge_hander, collection_name, feature_name, data_df, is_turn_to_id=True):
    '''
    用户ID和特征之间建立关系
    :param edge_hander:
    :param collection_name:
    :param feature_name:
    :param data_df:
    :param is_turn_to_id:
    :return:
    '''
    result = []
    if is_turn_to_id:
        feature_id_dic = get_dict_id(feature_name)
        for index, row in data_df.iterrows():
            dic = {
                '_from': "%s/%s" % (collection_name, feature_id_dic[row[feature_name]]),
                '_to': "CustomIds/" + row['customNum']
            }
            # result.append(dic)
            edge_hander.insert(dic)
    else:

        for index, row in data_df.iterrows():
            dic = {
                '_from': "%s/%s" % (collection_name, row[feature_name]),
                '_to': "CustomIds/" + row['customNum']
            }
            # result.append(dic)
            try:
                edge_hander.insert(dic)
            except:
                print(dic)


def create_relation_custom_and_feature_and_relation(edge_hander, collection_name, feature_name, relation_feature_name,
                                                    data_df, is_turn_to_id=True):
    '''
    用户ID和特征之间建立关系,并且关系由关系特征提供
    :param edge_hander:
    :param collection_name:
    :param feature_name:
    :param relation_feature_name:
    :param data_df:
    :param is_turn_to_id:
    :return:
    '''
    result = []
    if is_turn_to_id:
        feature_id_dic = get_dict_id(feature_name)
        for index, row in data_df.iterrows():
            dic = {
                '_from': "%s/%s" % (collection_name, feature_id_dic[row[feature_name]]),
                '_to': "CustomIds/" + row['customNum'],
                'relation': row[relation_feature_name]
            }
            # result.append(dic)
            edge_hander.insert(dic)
    else:

        for index, row in data_df.iterrows():
            dic = {
                '_from': "%s/%s" % (collection_name, row[feature_name]),
                '_to': "CustomIds/" + row['customNum'],
                'relation': row[relation_feature_name]
            }
            # result.append(dic)
            try:
                edge_hander.insert(dic)
            except:
                print(dic)


clinet = ArangoClient(host="10.111.32.65")
# db = clinet.db('financerisk', username="root", password="hxixpLi")
db = clinet.db('financerisk', username="hushiwei", password="hushiwei")

data_df = get_df("apply_nomissing.csv")

####################################################################################################

# 需要先转id的
# 与用户ID关联的
# collections_names = ['WorkName','WorkAddr','ContactAddr','HomeAddr', 'IdAddr', 'HouseAddr']
# feature_names = ['workName','workAddr','contactAddr','homeAddr', 'idAddr', 'houseAddr']
#
# for i in range(6):
#     print('~' * 10, 'start create relation : between [ CustomIds & %s ]' % (collections_names[i]), '~' * 10)
#     from_to_df = get_custom_and_feature_df(feature_names[i], data_df)
#     graph = create_graph_tool(db, "graph_" + collections_names[i])
#     edge_hander = create_edge_hander(graph, collections_names[i])
#     create_relation_custom_and_feature(edge_hander, collections_names[i], feature_names[i], from_to_df)
#     delete_graph(db, "graph_" + collections_names[i])

####################################################################################################

# 直接入库的
# 与用户ID关联的
# collections_names = ['CustomIds', 'Phone', 'MobilePhone']
# feature_names = ['customNum', 'phone', 'mobilePhone']
# for i in range(0, 4):
#     print('~' * 10, 'start create relation : between [ CustomIds & %s ]' % (collections_names[i]), '~' * 10)
#     from_to_df = get_custom_and_feature_df(feature_names[i], data_df)
#     graph = create_graph_tool(db, "graph_" + collections_names[i])
#     edge_hander = create_edge_hander(graph, collections_names[i])
#     create_relation_custom_and_feature(edge_hander, collections_names[i], feature_names[i], from_to_df,
#                                        is_turn_to_id=False)
#     delete_graph(db, "graph_" + collections_names[i])

####################################################################################################

# RelationCustomNum 与 CustomIds 通过 AndCustomRelation 建立关系
# 与用户ID关联的
# 多了从一个关系列中获取值
# collections_names = ['RelationCustomNum']
# feature_names = ['relationCustomNum']
# for i in range(0, 1):
#     print('~' * 10, 'start create relation : between [ CustomIds & %s ]' % (collections_names[i]), '~' * 10)
#     from_to_relation_df = get_custom_and_feature_relation_df(feature_names[i], "AndCustomRelation", data_df)
#     graph = create_graph_tool(db, "graph_" + collections_names[i])
#     edge_hander = create_edge_hander(graph, collections_names[i])
#     create_relation_custom_and_feature_and_relation(edge_hander, collections_names[i], feature_names[i],
#                                                     "AndCustomRelation", from_to_relation_df,
#                                                     is_turn_to_id=False)
#     delete_graph(db, "graph_" + collections_names[i])

####################################################################################################

# 两两之间from->to
# 联系人与联系人自己的电话
# collections_names = ['RelationMobile', 'RelationCustomNum']
# feature_names = ['relationMobile', 'relationCustomNum']
# print('~' * 10, 'start create relation : between [ %s & %s ]' % (collections_names[0], collections_names[1]), '~' * 10)
# from_to_df = get_from_and_to_df(feature_names[0], feature_names[1],data_df)
# graph = create_graph_tool(db, "graph_" + collections_names[0] + "_" + collections_names[1])
# edge_hander = create_edge_hander(graph, collections_names[0], collections_names[1])
# create_relation_feature_and_feature(edge_hander, collections_names, feature_names,
#                                     from_to_df, is_turn_to_id=False)
# delete_graph(db, "graph_" + collections_names[0] + "_" + collections_names[1])

# 工作单位->工作地址 并且需要转ID
collections_names = ['WorkNames', 'WorkAddrs']
feature_names = ['workName', 'workAddr']
print('~' * 10, 'start create relation : between [ %s & %s ]' % (collections_names[0], collections_names[1]), '~' * 10)
from_to_df = get_from_and_to_df(feature_names[0], feature_names[1], data_df)
graph = create_graph_tool(db, "graph_" + collections_names[0] + "_" + collections_names[1])
edge_hander = create_edge_hander(graph, collections_names[0], collections_names[1])
create_relation_feature_and_feature(edge_hander, collections_names, feature_names, from_to_df)
delete_graph(db, "graph_" + collections_names[0] + "_" + collections_names[1])
