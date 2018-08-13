# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/8/13
    Desc : IDCNN(Iterated Dilated CNN) 膨胀卷积神经网络
    Note : 
'''

'''
膨胀卷积的好处是不做pooling损失信息的情况下，加大了感受野，让每个卷积输出都包含较大范围的信息。在图像需要全局信息或者自然语言处理中需要较长的sequence信息依赖的问题中，都能很好的应用。
tf.nn.atrous_conv2d(value,filters,rate,padding,name=None）
value：输入的卷积图像，[batch, height, width, channels]。
filters：卷积核，[filter_height, filter_width, channels, out_channels]，通常NLP相关height设为1。
rate：正常的卷积通常会有stride，即卷积核滑动的步长，而膨胀卷积通过定义卷积和当中穿插的rate-1个0的个数，实现对原始数据采样间隔变大。
padding：”SAME”：补零   ； ”VALID”：丢弃多余的
'''

'''
模型是4个大的相同结构的Dilated CNN block拼在一起，每个block里面是dilation width为1, 1, 2的三层Dilated卷积层，所以叫做 Iterated Dilated CNN
'''
import tensorflow as tf

layers = [
    {
        'dilation': 1
    },
    {
        'dilation': 1
    },
    {
        'dilation': 2
    },
]
finalOutFromLayers = []
totalWidthForLastDim = 0
for j in range(4):
    for i in range(len(layers)):
        dilation = layers[i]['dilation']
        isLast = True if i == (len(layers) - 1) else False
        w = tf.get_variable("filterW", shape=[1, filter_width, num_filter, num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("filterB", shape=[num_filter])
        conv = tf.nn.atrous_conv2d(layerInput, w, rate=dilation, padding="SAME")
        conv = tf.nn.bias_add(conv, b)
        conv = tf.nn.relu(conv)
        if isLast:
            finalOutFromLayers.append(conv)
            totalWidthForLastDim += num_filter
        layerInput = conv
finalOut = tf.concat(axis=3, values=finalOutFromLayers)
