# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019/6/19
    Desc : 
    Note : bert-serving-start -model_dir /tmp/chinese_L-12_H-768_A-12 -num_worker=4
'''
from bert_serving.client import BertClient
import numpy as np


def cosine(a, b):
    return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


bc = BertClient()
# arr=bc.encode(['First do it', 'then do it right', 'then do it better'])



def two_word(name1, name2):
    emb = np.array(bc.encode([name1, name2]))
    print([name1, name2], ":", cosine(emb[0], emb[1]))


two_word('集奥聚合人工智能有限公司', '集奥聚合信息科技有限公司')
two_word('集奥聚合人工智能有限公司', '阿里巴巴国际信息科技有限公司')
two_word('集奥聚合人工智能有限公司', '国际信息科技有限公司')
two_word('集奥聚合信息科技有限公司', '金电联行信息科技有限公司')

print('`'*100)
two_word('集奥聚合人工智能', '集奥聚合信息科技')
two_word('集奥聚合人工智能有限公司', '阿里巴巴国际信息科技有限公司')
two_word('集奥聚合人工智能有限公司', '国际信息科技有限公司')
two_word('集奥聚合', '金电联行')