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
# read
with open("./workname_id.json", "r") as fp:
    workname_id=json.load(fp)

print(workname_id['南京豪威冰箱空调商场第二豪威交家电商场'])






from arango import ArangoClient

clinet = ArangoClient(host="10.111.32.65")
# db = clinet.db('financerisk', username="root", password="hxixpLi")
db = clinet.db('financerisk', username="hushiwei", password="hushiwei")

if db.has_graph('l0'):
    graph = db.graph('l0')
else:
    graph = db.create_graph('l0')

register = graph.create_edge_definition(
    edge_collection='customeid_workName',
    from_vertex_collections=['WorkNames'],
    to_vertex_collections=['CustomIds']
)

print(register.edges('WorkNames/798', direction='out'))

# Traverse the graph in outbound direction, breadth-first.
result = graph.traverse(
    start_vertex='WorkNames/798',
    direction='outbound',
    strategy='breadthfirst'
)