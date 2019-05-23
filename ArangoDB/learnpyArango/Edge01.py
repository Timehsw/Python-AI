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



clinet = ArangoClient(host="10.111.25.138")
db = clinet.db('GameOfThrones', username=":role:hushiwei", password="hushiwei")

#
# if db.has_collection('customeid_mobilephone'):
#     students = db.create_graph('customeid_mobilephone')
# else:
#     students = db.create_graph('customeid_mobilephone')
#
# oneline={
#     '_from': 'CustomIds/3D0B57F2C682A67E42EF8567C43ADF0C',
#     '_to': 'MobilePhone/13804562715',
# }
#
# metadata = students.insert(oneline)
# students.

# mobile_phone=db.collection("MobilePhone")
# print(mobile_phone.get('13804568133'))
#
#
#
# a=db.collection("ChildOf")
# print(a.get('192214'))

# Create a new graph named "school".
graph = db.create_graph('adfdsdgdsgfa')

# # Create vertex collections for the graph.
#
# # Create an edge definition (relation) for the graph.
register = graph.create_edge_definition(
    edge_collection='customeid_mobilephone',
    from_vertex_collections=['MobilePhone'],
    to_vertex_collections=['CustomIds']
)
#
#
# # Insert edge documents into "register" edge collection.

# register.insert({'_from': 'MobilePhone/13804562715', '_to': 'CustomIds/3D0B57F2C682A67E42EF8567C43ADF0C'})
print(register.edges('MobilePhone/13804562715', direction='out'))

# Traverse the graph in outbound direction, breadth-first.
# result = graph.traverse(
#     start_vertex='MobilePhone/13804562715',
#     direction='outbound',
#     strategy='breadthfirst'
# )

