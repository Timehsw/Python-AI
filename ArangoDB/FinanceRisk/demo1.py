# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-17
    Desc : 
    Note : 
'''
import json
import os

root_path = "/Users/hushiwei/Downloads/模拟数据"
os.chdir(root_path)
# # read
# with open("./workname_id.json", "r") as fp:
#     workname_id=json.load(fp)
#
# print(workname_id['南京豪威冰箱空调商场第二豪威交家电商场'])

aa="hello/%s/%s"%(1,3)
print(aa)




from arango import ArangoClient

clinet = ArangoClient(host="10.111.32.65")
# db = clinet.db('financerisk', username="root", password="hxixpLi")
db = clinet.db('financerisk', username="hushiwei", password="hushiwei")

if db.has_graph('l2'):
    graph = db.graph('l2')
else:
    graph = db.create_graph('l2')

# register = graph.create_edge_definition(
#     edge_collection='customeid_workName',
#     from_vertex_collections=['WorkNames'],
#     to_vertex_collections=['CustomIds']
# )
#
# print(register.edges('WorkNames/798', direction='out'))

# Traverse the graph in outbound direction, breadth-first.
result = graph.traverse(
    start_vertex='CustomIds/3D0B57F2C682A67E42EF8567C43ADF0C',
    direction='any',
    strategy='breadthfirst',
    min_depth=1
)