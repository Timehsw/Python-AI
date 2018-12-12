# 《机器学习》一书中涉及到的代码

## 1.线性模型

### 1.1对数几率回归  [📎](./LinerModel/LogisticRegression.py)

原题在西瓜书P69页3.3

![LogisticRe](./Img/Logistic_Regression.png)

## 2.决策树 [📎](./DecisionTree/decisionTree.py)

完成了两种评价方式与三种剪枝策略的离散决策树的编写，评价函数对决策树的影响有限。PS：连续值的决策树与多变量决策树有点难，脑壳疼不想写😋

![DT](./Img/DecisionTree.png)

## 3.神经网络

### 3.1.累积形BP神经网络 [📎](./NeuralNet/BP_Tensorflow.py)

针对书上的西瓜数据集3.0，用BP神经网络进行了拟合，最好结果为：
学习率为0.05,隐含层维度为8,最终测试集误差为0.12090,训练了4330次,正确率为:1.00000
ps:我就是传说中的调参工程师

![BP](./Img/bpnn_structure.png)

### 3.2.RBF神经网络 [📎](./NeuralNet/RBFnn.py)

用RBF网络实现了异或操作，中间层4阶，学习率0.1，迭代1k次结果为：

* 0 xor 0 is 0.025424
* 0 xor 1 is 0.938498
* 1 xor 0 is 0.868398
* 1 xor 1 is 0.163135

## 4.支持向量机（SVM）

### 4.1 线性核与高斯核SVM在西瓜3.0α上的分类结果 [📎](./SVM/svm_train.py)

准确率明显高斯核（'-s 0 -t 2 -c 1000'）比线性核（'-s 0 -t 0 -c 100'）高

![SVM](./Img/liner_Gaussian_SVM.png)

## 5.集成学习

### 5.1 AdaBoost [📎](./EnsembleLearning/AdaBoost.py)

使用AdaBoost对决策树桩进行提升生成分类器。在西瓜3.0α训练集上的效果(8轮）如图

![AdaBoost](./Img/AdaBoosting.png)

## 6.K近邻算法  [📎](./KNearestNeighbor/KNearestNeighbor.py)

采用KD树检索样本点

![KNN](./Img/KNN.png)
