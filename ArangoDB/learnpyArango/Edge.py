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
graph = db.create_graph('school')

# Create vertex collections for the graph.
students = graph.create_vertex_collection('students')
lectures = graph.create_vertex_collection('lectures')

# Create an edge definition (relation) for the graph.
register = graph.create_edge_definition(
    edge_collection='register',
    from_vertex_collections=['students'],
    to_vertex_collections=['lectures']
)

# Insert vertex documents into "students" (from) vertex collection.
students.insert({'_key': '01', 'full_name': 'Anna Smith'})
students.insert({'_key': '02', 'full_name': 'Jake Clark'})
students.insert({'_key': '03', 'full_name': 'Lisa Jones'})

# Insert vertex documents into "lectures" (to) vertex collection.
lectures.insert({'_key': 'MAT101', 'title': 'Calculus'})
lectures.insert({'_key': 'STA101', 'title': 'Statistics'})
lectures.insert({'_key': 'CSC101', 'title': 'Algorithms'})

# Insert edge documents into "register" edge collection.
register.insert({'_from': 'students/01', '_to': 'lectures/MAT101'})
register.insert({'_from': 'students/01', '_to': 'lectures/STA101'})
register.insert({'_from': 'students/01', '_to': 'lectures/CSC101'})
register.insert({'_from': 'students/02', '_to': 'lectures/MAT101'})
register.insert({'_from': 'students/02', '_to': 'lectures/STA101'})
register.insert({'_from': 'students/03', '_to': 'lectures/CSC101'})

# Traverse the graph in outbound direction, breadth-first.
result = graph.traverse(
    start_vertex='students/01',
    direction='outbound',
    strategy='breadthfirst'
)