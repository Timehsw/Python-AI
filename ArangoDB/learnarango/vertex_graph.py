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

# Create a new vertex collection named "teachers" if it does not exist.
# This returns an API wrapper for "teachers" vertex collection.
if school.has_vertex_collection('teachers'):
    teachers = school.vertex_collection('teachers')
else:
    teachers = school.create_vertex_collection('teachers')

# List vertex collections in the graph.
school.vertex_collections()

# Vertex collections have similar interface as standard collections.
teachers.properties()
teachers.insert({'_key': 'jon', 'name': 'Jon'})
teachers.update({'_key': 'jon', 'age': 35})
teachers.replace({'_key': 'jon', 'name': 'Jon', 'age': 36})
teachers.get('jon')
teachers.has('jon')
teachers.delete('jon')

#########################################################################

# Initialize the ArangoDB client.
client = ArangoClient()

# Connect to "test" database as root user.
db = client.db('test', username='root', password='passwd')

# Get the API wrapper for graph "school".
school = db.graph('school')

# Create a new vertex collection named "lectures" if it does not exist.
# This returns an API wrapper for "lectures" vertex collection.
if school.has_vertex_collection('lectures'):
    school.vertex_collection('lectures')
else:
    school.create_vertex_collection('lectures')

# The "_id" field is required instead of "_key" field (except for insert).
school.insert_vertex('lectures', {'_key': 'CSC101'})
school.update_vertex({'_id': 'lectures/CSC101', 'difficulty': 'easy'})
school.replace_vertex({'_id': 'lectures/CSC101', 'difficulty': 'hard'})
school.has_vertex('lectures/CSC101')
school.vertex('lectures/CSC101')
school.delete_vertex('lectures/CSC101')