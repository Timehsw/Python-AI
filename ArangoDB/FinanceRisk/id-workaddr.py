# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : 
    Note : 
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

clinet = ArangoClient(host="10.111.32.65")
# db = clinet.db('financerisk', username="root", password="hxixpLi")
db = clinet.db('financerisk', username="hushiwei", password="hushiwei")

# Create a new graph named "school".
graph = db.create_graph('l1')

register = graph.create_edge_definition(
    edge_collection='customeid_workAddr',
    from_vertex_collections=['WorkAddrs'],
    to_vertex_collections=['CustomIds']
)


root_path = "/Users/hushiwei/Downloads/模拟数据"
os.chdir(root_path)

apply_path = "apply.csv"
apply_df = pd.read_csv(apply_path, error_bad_lines=False)

print(apply_df)

# 获取公司名称,去除中文中的数字
apply_df['workName'] = apply_df['workName'].str.extract(u"([\u4e00-\u9fa5]+)", expand=False)
data = apply_df[["customNum", "workName"]]
print(data)

# read
with open("./workname_id.json", "r") as fp:
    workname_id=json.load(fp)


result = []
for index, row in data.iterrows():
    dic = {
        '_from': "WorkName/" + workname_id[row['workName']],
        '_to': "CustomIds/" + row['customNum']
    }
    # result.append(dic)
    register.insert(dic)
# register.insert({'_from': 'WorkName/13804562715', '_to': 'CustomIds/3D0B57F2C682A67E42EF8567C43ADF0C'})
