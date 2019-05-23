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

# The "_id" field is required instead of "_key" field.
school.insert_edge(
    collection='teach',
    edge={
        '_id': 'teach/jon-CSC101',
        '_from': 'teachers/jon',
        '_to': 'lectures/CSC101'
    }
)
school.replace_edge({
    '_id': 'teach/jon-CSC101',
    '_from': 'teachers/jon',
    '_to': 'lectures/CSC101',
    'online': False,
})
school.update_edge({
    '_id': 'teach/jon-CSC101',
    'online': True
})
school.has_edge('teach/jon-CSC101')
school.edge('teach/jon-CSC101')
school.delete_edge('teach/jon-CSC101')
school.link('teach', 'teachers/jon', 'lectures/CSC101')
school.edges('teach', 'teachers/jon', direction='in')