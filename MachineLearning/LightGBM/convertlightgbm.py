# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-02-13
    Desc : 
    Note : 读取json文件进行
'''
import numpy as np
import pandas as pd
import json

class InferenceLightGBM(object):
    '''
    用于lightGBM dump_model产生的模型进行推断，以实现在没有安装lightGBM的电脑上进行解析模型
    '''

    def __init__(self, model_file=None, category_file=None):

        with open(model_file, 'r') as json_file:
            self.model_json = json.load(json_file)
            # 模型json 字典
        with open(category_file, 'r') as json_file:
            # 分类特征序号
            self.categories = json.load(json_file)
        #             print(self.categories)

        self.feature_names = self.model_json['feature_names']

    def predict(self, X):
        '''
        预测样本
        '''
        try:
            columns = list(X.columns)
        except:
            print('{} should be a pandas.DataFrame'.format(X))

        if self.model_json['feature_names'] == columns:
            y = self._predict(X)
            return y
        else:
            raise Exception("columns should be {}".format(self.feature_names), )

    def _sigmoid(self, z):

        return 1.0 / (1 + np.exp(-z))

    def _predict(self, X):
        '''
        对模型树字典进行解析
        '''
        feat_names = self.feature_names
        results = pd.Series(index=X.index)
        trees = self.model_json['tree_info']
        for idx in X.index:
            X_sample = X.loc[idx:idx, :]
            leaf_values = 0.0
            # 对不同的树进行循环
            for tree in trees:
                tree_structure = tree['tree_structure']
                leaf_value = self._walkthrough_tree(tree_structure, X_sample)
                leaf_values += leaf_value
            results[idx] = self._sigmoid(leaf_values)
        return results

    def _walkthrough_tree(self, tree_structure, X_sample):
        '''
        递归式对树进行遍历，返回最后叶子节点数值
        '''
        if 'leaf_index' in tree_structure.keys():
            # 此时已到达叶子节点
            return tree_structure['leaf_value']
        else:
            # 依然处于分裂点
            split_feature = X_sample.iloc[0, tree_structure['split_feature']]
            decision_type = tree_structure['decision_type']
            threshold = tree_structure['threshold']

            # 类别特征
            if decision_type == '==':
                feat_name = self.feature_names[tree_structure['split_feature']]
                categories = self.categories[feat_name]
                category = categories[str(split_feature)]
                category = str(category)
                threshold = threshold.split('||')
                if category in threshold:
                    tree_structure = tree_structure['left_child']
                else:
                    tree_structure = tree_structure['right_child']
                return self._walkthrough_tree(tree_structure, X_sample)
            # 数值特征
            elif decision_type == '<=':
                if split_feature <= threshold:
                    tree_structure = tree_structure['left_child']
                else:
                    tree_structure = tree_structure['right_child']

                return self._walkthrough_tree(tree_structure, X_sample)
            else:
                print(tree_structure)
                print('decision_type: {} is not == or <='.format(decision_type))
                return None
