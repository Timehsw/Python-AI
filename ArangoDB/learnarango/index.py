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

# Create a new collection named "cities".
cities = db.create_collection('cities')

# List the indexes in the collection.
cities.indexes()

# Add a new hash index on document fields "continent" and "country".
index = cities.add_hash_index(fields=['continent', 'country'], unique=True)

# Add new fulltext indexes on fields "continent" and "country".
index = cities.add_fulltext_index(fields=['continent'])
index = cities.add_fulltext_index(fields=['country'])

# Add a new skiplist index on field 'population'.
index = cities.add_skiplist_index(fields=['population'], sparse=False)

# Add a new geo-spatial index on field 'coordinates'.
index = cities.add_geo_index(fields=['coordinates'])

# Add a new persistent index on fields 'currency'.
index = cities.add_persistent_index(fields=['currency'], sparse=True)

# Delete the last index from the collection.
cities.delete_index(index['id'])