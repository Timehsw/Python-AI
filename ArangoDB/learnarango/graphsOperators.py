# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : 
    Note : 
'''

from arango import ArangoClient

# Initialize the client for ArangoDB.
client = ArangoClient(protocol='http', host='localhost', port=8529)

# Connect to "test" database as root user.
db = client.db('test', username='root', password='passwd')

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