# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019/6/19
    Desc : 
    Note : bert-serving-start -model_dir /tmp/chinese_L-12_H-768_A-12 -num_worker=4
'''
from bert_serving.client import BertClient
import numpy as np
import pandas as pd
from tqdm import tqdm

def cosine(a, b):
    return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


bc = BertClient()


# arr=bc.encode(['First do it', 'then do it right', 'then do it better'])


def two_word(name1, name2):
    emb = np.array(bc.encode([name1, name2]))
    print([name1, name2], ":", cosine(emb[0], emb[1]))


# two_word('集奥聚合人工智能有限公司', '集奥聚合信息科技有限公司')
# two_word('集奥聚合人工智能有限公司', '阿里巴巴国际信息科技有限公司')
# two_word('集奥聚合人工智能有限公司', '国际信息科技有限公司')
# two_word('集奥聚合信息科技有限公司', '金电联行信息科技有限公司')
#
# print('`'*100)
# two_word('集奥聚合人工智能', '集奥聚合信息科技')
# two_word('集奥聚合人工智能有限公司', '阿里巴巴国际信息科技有限公司')
# two_word('集奥聚合人工智能有限公司', '国际信息科技有限公司')
# two_word('集奥聚合', '金电联行')

path = '/mnt/d/地址/NLp/deal/poi_area_flat.txt'
df = pd.read_csv(path, sep='\t')

print(df.head())

selected = ['province', 'city', 'district']
df1 = df[selected]
df2 = df1['province'] + df1['city'] + df1['district']

df2.dropna(inplace=True)

address = [i[1] for i in df2.iteritems()]
print(address)
result_dic={}
# for i in tqdm(range(len(address))):
#     result = bc.encode([address[i]])
#     result.setflags(write=1)
#     result[address[i]]=str(result[0])
#     print(result)
print(df2.shape)
# df=pd.DataFrame(result_dic,columns=['name','encode'])
# print(result)
df2['encode']=df2.apply(lambda x:bc.encode([x]))
print(df2)
df2.to_csv("./adds.csv",index=False)
