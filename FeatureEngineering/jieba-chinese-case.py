# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/28
    Desc : jieba中文案例
'''

import warnings
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

jieba.add_word('郑秋冬')
jieba.add_word("离境天")
jieba.suggest_freq("离境天", True)

print('-' * 20, "对文档进行分词操作", '-' * 20)
with open('./datas/nlp_1.txt', encoding='utf-8') as f:
    document = f.read()
    document_cut = jieba.cut(document)
    result = " ".join(document_cut)
    with open('./datas/nlp_1_cut.txt', 'w', encoding='utf-8') as f2:
        f2.write(result)
# 第二个文档分词#
with open('./datas/nlp_2.txt', encoding='utf-8') as f:
    document = f.read()
    document_cut = jieba.cut(document)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document_cut)
    with open('./datas/nlp_2_cut.txt', 'w', encoding='utf-8') as f2:
        f2.write(result)
# 第三个文档分词#
with open('./datas/nlp_3.txt', encoding='utf-8') as f:
    document = f.read()
    jieba.load_userdict("./datas/mydict.txt")
    document_cut = jieba.cut(document)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document_cut)
    with open('./datas/nlp_3_cut.txt', 'w', encoding='utf-8') as f2:
        f2.write(result)

print('-' * 20, "读取分词后的文档", '-' * 20)

with open('./datas/nlp_1_cut.txt', encoding='utf-8') as f3:
    cut1 = f3.read()
# print(cut1)
with open('./datas/nlp_2_cut.txt', encoding='utf-8') as f4:
    cut2 = f4.read()
# print(cut2)
with open('./datas/nlp_3_cut.txt', encoding='utf-8') as f5:
    cut3 = f5.read()

print(cut3)

print()
print('-' * 20, "从文件导入停用词表", '-' * 20)

stpwrdpath = "./datas/stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
# 将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

corpus = [cut1, cut2, cut3]
CV = CountVectorizer(stop_words=stpwrdlst)
CTF = CV.fit_transform(corpus)
print(CTF.toarray())
words = CV.get_feature_names()

lda = LatentDirichletAllocation(n_topics=2, learning_method='batch',
                                learning_offset=50.,
                                random_state=0)
lda_result = lda.fit_transform(CTF)

print(lda_result)
print(lda.components_)
