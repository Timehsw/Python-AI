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

# Create some test documents to play around with.
# The documents must have the "_id" field instead.
lola = {'_id': 'students/lola', 'GPA': 3.5}
abby = {'_id': 'students/abby', 'GPA': 3.2}
john = {'_id': 'students/john', 'GPA': 3.6}
emma = {'_id': 'students/emma', 'GPA': 4.0}

# Insert a new document.
metadata = db.insert_document('students', lola)
assert metadata['_id'] == 'students/lola'
assert metadata['_key'] == 'lola'

# Check if a document exists.
assert db.has_document(lola) is True

# Get a document (by ID or body with "_id" field).
db.document('students/lola')
db.document(abby)

# Update a document.
lola['GPA'] = 3.6
db.update_document(lola)

# Replace a document.
lola['GPA'] = 3.4
db.replace_document(lola)

# Delete a document (by ID or body with "_id" field).
db.delete_document('students/lola')