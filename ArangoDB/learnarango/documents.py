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

# Get the API wrapper for "students" collection.
students = db.collection('students')

# Create some test documents to play around with.
lola = {'_key': 'lola', 'GPA': 3.5, 'first': 'Lola', 'last': 'Martin'}
abby = {'_key': 'abby', 'GPA': 3.2, 'first': 'Abby', 'last': 'Page'}
john = {'_key': 'john', 'GPA': 3.6, 'first': 'John', 'last': 'Kim'}
emma = {'_key': 'emma', 'GPA': 4.0, 'first': 'Emma', 'last': 'Park'}

# Insert a new document. This returns the document metadata.
metadata = students.insert(lola)
assert metadata['_id'] == 'students/lola'
assert metadata['_key'] == 'lola'

# Check if documents exist in the collection in multiple ways.
assert students.has('lola') and 'john' not in students

# Retrieve the total document count in multiple ways.
assert students.count() == len(students) == 1

# Insert multiple documents in bulk.
students.import_bulk([abby, john, emma])

# Retrieve one or more matching documents.
for student in students.find({'first': 'John'}):
    assert student['_key'] == 'john'
    assert student['GPA'] == 3.6
    assert student['last'] == 'Kim'

# Retrieve a document by key.
students.get('john')

# Retrieve a document by ID.
students.get('students/john')

# Retrieve a document by body with "_id" field.
students.get({'_id': 'students/john'})

# Retrieve a document by body with "_key" field.
students.get({'_key': 'john'})

# Retrieve multiple documents by ID, key or body.
students.get_many(['abby', 'students/lola', {'_key': 'john'}])

# Update a single document.
lola['GPA'] = 2.6
students.update(lola)

# Update one or more matching documents.
students.update_match({'last': 'Park'}, {'GPA': 3.0})

# Replace a single document.
emma['GPA'] = 3.1
students.replace(emma)

# Replace one or more matching documents.
becky = {'first': 'Becky', 'last': 'Solis', 'GPA': '3.3'}
students.replace_match({'first': 'Emma'}, becky)

# Delete a document by key.
students.delete('john')

# Delete a document by ID.
students.delete('students/lola')

# Delete a document by body with "_id" or "_key" field.
students.delete(emma)

# Delete multiple documents. Missing ones are ignored.
students.delete_many([abby, 'john', 'students/lola'])

# Iterate through all documents and update individually.
for student in students:
    student['GPA'] = 4.0
    student['happy'] = True
    students.update(student)