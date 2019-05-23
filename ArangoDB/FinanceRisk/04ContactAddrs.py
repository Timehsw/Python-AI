# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : ContactAddrs
    Note : ContactAddrs
'''

import os
import pandas as pd
import sys

from arango import ArangoClient
import numpy as np

def create_save_collections(featureColleciton, collection_name, df):
    feature_df = df[collection_name]
    feature_df.drop_duplicates(inplace=True)

    result = [{'_key': str(index), 'name': work_name} for index, work_name in feature_df.iteritems()]

    feature_id = {work_name: str(index) for index, work_name in feature_df.iteritems()}
    import json

    # write
    with open("./%s_id.json" % (collection_name), "w") as fp:
        json.dump(feature_id, fp)
    # read
    # with open("./workname_id.json", "r") as fp:
    #     workname_id=json.load(fp)

    featureColleciton.insert_many(result)


clinet = ArangoClient(host="10.111.32.65")
# db = clinet.db('financerisk', username="root", password="hxixpLi")
db = clinet.db('financerisk', username="hushiwei", password="hushiwei")
print(db.collections())
if db.has_collection("ContactAddrs"):
    featureColleciton = db.collection("ContactAddrs")
else:
    featureColleciton = db.create_collection("ContactAddrs")

root_path = "/Users/hushiwei/Downloads/模拟数据"
os.chdir(root_path)

apply_path = "apply_nomissing.csv"
apply_df = pd.read_csv(apply_path, error_bad_lines=False)

print(apply_df)

create_save_collections(featureColleciton, "contactAddr", apply_df)
