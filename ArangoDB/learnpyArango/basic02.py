# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : 
    Note : 
'''

from pyArango.connection import *
import os
import pandas as pd

conn = Connection(arangoURL='http://10.111.25.138:8529', username=":role:hushiwei", password="hushiwei", )

# db=conn.createDatabase(name="GameOfThrones")
db = conn['GameOfThrones']
print(db)

# customidCollection = db.createCollection(name="MobilePhone")
customidCollection = db["MobilePhone"]
doc = customidCollection.createDocument()
doc["name"] = "Nothing"
doc._key = str("Nothing")
doc.save()
conn.create
