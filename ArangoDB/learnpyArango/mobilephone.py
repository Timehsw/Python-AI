# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : 
    Note : 
'''

from pyArango.connection import *
import os
import pandas as pd
import sys
root_path = "/Users/hushiwei/Downloads/模拟数据"
os.chdir(root_path)

apply_path = "apply.csv"
apply_df = pd.read_csv(apply_path, error_bad_lines=False)

print(apply_df)
# customid=apply_df["customNum"].values.tolist()
conn = Connection(arangoURL='http://10.111.25.138:8529', username=":role:hushiwei", password="hushiwei", )

# db=conn.createDatabase(name="GameOfThrones")
db = conn['GameOfThrones']
print(db)

# customidCollection = db.createCollection(name="MobilePhone")
customidCollection = db['MobilePhone']
print(apply_df["mobilePhone"].shape)
apply_df["mobilePhone"].dropna(inplace=True)
apply_df["mobilePhone"].drop_duplicates(inplace=True)
print(apply_df["mobilePhone"].shape)

for phone in apply_df["mobilePhone"] :
    if phone is not None:
        try:
            doc = customidCollection.createDocument()
            doc["name"] = phone
            doc._key = str(phone)
            doc.save()

        except:
            print(phone)
            sys.exit()

