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
graph = db.create_graph('l')

register = graph.create_edge_definition(
    edge_collection='customeid_workName',
    from_vertex_collections=['WorkNames'],
    to_vertex_collections=['CustomIds']
)


root_path = "/Users/hushiwei/Downloads/模拟数据"
os.chdir(root_path)

apply_path = "apply_nomissing.csv"
apply_df = pd.read_csv(apply_path, error_bad_lines=False)

print(apply_df)

data = apply_df[["customNum", "workName"]]
print(data)

# read
with open("./workname_id.json", "r") as fp:
    workname_id=json.load(fp)


result = []
for index, row in data.iterrows():
    dic = {
        '_from': "WorkNames/" + workname_id[row['workName']],
        '_to': "CustomIds/" + row['customNum']
    }
    # result.append(dic)
    register.insert(dic)
# register.insert({'_from': 'WorkName/13804562715', '_to': 'CustomIds/3D0B57F2C682A67E42EF8567C43ADF0C'})
