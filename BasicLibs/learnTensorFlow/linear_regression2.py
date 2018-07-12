# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/11
    Desc : tensorflow 之 梯度下降求解线性回归 | 分布式执行
'''


import numpy as np
import tensorflow as tf

np.random.seed(28)

# TODO： 将这个代码整理成为单机运行的

# 1. 配置服务器相关信息
# 因为tensorflow底层代码中，默认就是使用ps和work分别表示两类不同的工作节点
# ps：变量/张量的初始化、存储相关节点
# work: 变量/张量的计算/运算的相关节点
ps_hosts = ['127.0.0.1:33331', '127.0.0.1:33332']
work_hosts = ['127.0.0.1:33333', '127.0.0.1:33334', '127.0.0.1:33335']
cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'work': work_hosts})

# 2. 定义一些运行参数(在运行该python文件的时候就可以指定这些参数了)
tf.app.flags.DEFINE_integer('task_index', default_value=0, docstring="Index of task within the job")
FLAGS = tf.app.flags.FLAGS


# 3. 构建运行方法
def main(_):
    # 图的构建
    with tf.device(
            tf.train.replica_device_setter(worker_device='/job:work/task:%d' % FLAGS.task_index, cluster=cluster)):
        # 构建一个样本的占位符信息
        x_data = tf.placeholder(tf.float32, [10])
        y_data = tf.placeholder(tf.float32, [10])

        # 定义一个变量w和变量b
        # random_uniform：（random意思：随机产生数据， uniform：均匀分布的意思） ==> 意思：产生一个服从均匀分布的随机数列
        # shape: 产生多少数据/产生的数据格式是什么； minval：均匀分布中的可能出现的最小值，maxval: 均匀分布中可能出现的最大值
        w = tf.Variable(initial_value=tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0), name='w')
        b = tf.Variable(initial_value=tf.zeros([1]), name='b')
        # 构建一个预测值
        y_hat = w * x_data + b

        # 构建一个损失函数
        # 以MSE作为损失函数（预测值和实际值之间的平方和）
        loss = tf.reduce_mean(tf.square(y_hat - y_data), name='loss')

        global_step = tf.Variable(0, name='global_step', trainable=False)
        # 以随机梯度下降的方式优化损失函数
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
        # 在优化的过程中，是让那个函数最小化
        train = optimizer.minimize(loss, name='train', global_step=global_step)

    # 图的运行
    hooks = [tf.train.StopAtStepHook(last_step=10000000)]
    with tf.train.MonitoredTrainingSession(
            master='grpc://' + work_hosts[FLAGS.task_index],
            is_chief=(FLAGS.task_index == 0),  # 是否进行变量的初始化，设置为true，表示进行初始化
            checkpoint_dir='./tmp',
            save_checkpoint_secs=None,
            hooks=hooks  # 给定的条件
    ) as mon_sess:
        while not mon_sess.should_stop():
            N = 10
            train_x = np.linspace(0, 6, N) + np.random.normal(loc=0.0, scale=2, size=N)
            train_y = 14 * train_x - 7 + np.random.normal(loc=0.0, scale=5.0, size=N)
            _, step, loss_v, w_v, b_v = mon_sess.run([train, global_step, loss, w, b],
                                                     feed_dict={x_data: train_x, y_data: train_y})
            if step % 100 == 0:
                print('Step:{}, loss:{}, w:{}, b:{}'.format(step, loss_v, w_v, b_v))


if __name__ == '__main__':
    tf.app.run()