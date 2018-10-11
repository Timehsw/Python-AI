# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/11
    Desc : CountVectorizer
    Note : 
'''

from sklearn.feature_extraction.text import CountVectorizer

#语料
corpus = [
        'This is the first document.',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document?'
        ]

# 将文本中的词语，转换成词频矩阵

vectorizer = CountVectorizer()

# 计算词语出现的频率
X = vectorizer.fit_transform(corpus)

# 获取词袋中所有文本关键词
words = vectorizer.get_feature_names()
print(words)

# 查看词频结果
'''
同时在输出每个句子中包含特征词的个数。例如，第一句“This is the first document.”，它对应的
词频为[0, 1, 1, 1, 0, 0, 1, 0, 1]，假设初始序号从1开始计数，则该词频表示存在第2个位置的单
词“document”共1次、第3个位置的单词“first”共1次、第4个位置的单词“is”共1次、第9个位置的单词
“this”共1词。所以，每个句子都会得到一个词频向量。
'''
print(X.toarray())
