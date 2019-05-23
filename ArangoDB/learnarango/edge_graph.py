# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : 
    Note : 
'''
# Here is an example showing how edge definitions are managed:
from arango import ArangoClient

# Initialize the ArangoDB client.
client = ArangoClient()

# Connect to "test" database as root user.
db = client.db('test', username='root', password='passwd')

# Get the API wrapper for graph "school".
if db.has_graph('school'):
    school = db.graph('school')
else:
    school = db.create_graph('school')

# Create an edge definition named "teach". This creates any missing
# collections and returns an API wrapper for "teach" edge collection.
if not school.has_edge_definition('teach'):
    teach = school.create_edge_definition(
        edge_collection='teach',
        from_vertex_collections=['teachers'],
        to_vertex_collections=['teachers']
    )

# List edge definitions.
school.edge_definitions()

# Replace the edge definition.
school.replace_edge_definition(
    edge_collection='teach',
    from_vertex_collections=['teachers'],
    to_vertex_collections=['lectures']
)

# Delete the edge definition (and its collections).
school.delete_edge_definition('teach', purge=True)