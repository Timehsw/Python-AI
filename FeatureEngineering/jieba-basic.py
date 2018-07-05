# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/28
    Desc : jieba基础操作
    安装: pip install jieba
'''

import jieba

words_a = "上海自来水来自海上,所以吃葡萄不吐葡萄皮"

print('-' * 20, "jieba 的三种分词模式: cut", "-" * 20)

seg_a = jieba.cut(words_a, cut_all=True)
print("全模式: ", "/".join(seg_a))

seg_b = jieba.cut(words_a)
print("精确模式: ", "/".join(seg_b))

seg_c = jieba.cut_for_search(words_a)
print("搜索引擎模式: ", "/".join(seg_c))

# jieba.cut()返回的是一个迭代类型.上面已经迭代出来打印了
# 如果还需要打印,那么需要重新获取
for i in jieba.cut(words_a):
    print(i)

print('-' * 20, "添加和删除自定义词汇", "-" * 20)

words_a1 = "我为机器学习疯狂打call"
print("自定义前: ", "/".join(jieba.cut(words_a1)))
jieba.del_word("学习")  # 删除单词,在后续分词的时候,被删除的不会认为是一个单词
jieba.add_word("打call")  # 添加单词,在后续分词的时候,遇到的时候,会认为是属于同一个单词
print("删除'学习',加入'打call'后: ", "/".join(jieba.cut(words_a1)))

print('-' * 20, "导入自定义词典", "-" * 20)
jieba.del_word("打call")  # 删除之前添加的词汇
words_a2 = "在复仇者联盟2的电影里,钢铁侠和奇异博士们联手也干不过灭霸这个终极大boss,我高喊666,为编辑疯狂打call"
print('加载自定义词库前: ', "/".join(jieba.cut(words_a2)))
jieba.load_userdict('./datas/mydict.txt')
print("---------- VS -----------")
print("加载自定义词库后: ", "/".join(jieba.cut(words_a2)))

print('-' * 20, "lcut获得切词后的数据列表", "-" * 20)
ls1 = []
for item in jieba.cut(words_a2):
    ls1.append(item)
print(ls1)
# 用lcut直接获得切词后的list列表数据
ls2 = jieba.lcut(words_a2)
print(ls2)
print(type(ls2))
print("~" * 10)
ls3 = list(jieba.cut(words_a2))
print(ls3)
print(type(ls3))

print('-' * 20, "调整词典,关闭HMM发现新词功能(主要在开发过程中使用)", "-" * 20)
print("/".join(jieba.cut("如果放到post中将出错.", HMM=False)))
jieba.suggest_freq(('中', "将"), True)
print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
jieba.suggest_freq('台中', True)
print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))

print('-' * 20, "获取TF-IDF最大的20个单词", "-" * 20)
import jieba.analyse

tags = jieba.analyse.extract_tags(words_a2, topK=20, withWeight=True)
for tag in tags:
    print(tag)
