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

# Get the API wrapper for edge collection "teach".
if school.has_edge_definition('teach'):
    teach = school.edge_collection('teach')
else:
    teach = school.create_edge_definition(
        edge_collection='teach',
        from_vertex_collections=['teachers'],
        to_vertex_collections=['lectures']
    )

# Edge collections have a similar interface as standard collections.
teach.insert({
    '_key': 'jon-CSC101',
    '_from': 'teachers/jon',
    '_to': 'lectures/CSC101'
})
teach.replace({
    '_key': 'jon-CSC101',
    '_from': 'teachers/jon',
    '_to': 'lectures/CSC101',
    'online': False
})
teach.update({
    '_key': 'jon-CSC101',
    'online': True
})
teach.has('jon-CSC101')
teach.get('jon-CSC101')
teach.delete('jon-CSC101')

# Create an edge between two vertices (essentially the same as insert).
teach.link('teachers/jon', 'lectures/CSC101', data={'online': False})

# List edges going in/out of a vertex.
teach.edges('teachers/jon', direction='in')
teach.edges('teachers/jon', direction='out')