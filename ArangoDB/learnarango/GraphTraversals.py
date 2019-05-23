# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : 
    Note : 
'''

from arango import ArangoClient

# Initialize the ArangoDB client.
client = ArangoClient()

# Connect to "test" database as root user.
db = client.db('test', username='root', password='passwd')

# Get the API wrapper for graph "school".
school = db.graph('school')

# Get API wrappers for "from" and "to" vertex collections.
teachers = school.vertex_collection('teachers')
lectures = school.vertex_collection('lectures')

# Get the API wrapper for the edge collection.:
teach = school.edge_collection('teach')

# Insert vertices into the graph.
teachers.insert({'_key': 'jon', 'name': 'Professor jon'})
lectures.insert({'_key': 'CSC101', 'name': 'Introduction to CS'})
lectures.insert({'_key': 'MAT223', 'name': 'Linear Algebra'})
lectures.insert({'_key': 'STA201', 'name': 'Statistics'})

# Insert edges into the graph.
teach.insert({'_from': 'teachers/jon', '_to': 'lectures/CSC101'})
teach.insert({'_from': 'teachers/jon', '_to': 'lectures/STA201'})
teach.insert({'_from': 'teachers/jon', '_to': 'lectures/MAT223'})

# Traverse the graph in outbound direction, breath-first.
school.traverse(
    start_vertex='teachers/jon',
    direction='outbound',
    strategy='bfs',
    edge_uniqueness='global',
    vertex_uniqueness='global',
)