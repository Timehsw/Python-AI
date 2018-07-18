# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/18
    Desc : 图片处理相关API
    备注 : 图像处理的python库:opencv,PIL,matplotlib,tensorflow等
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 打印numpy的数组对象的时候,中间不省略
np.set_printoptions(threshold=np.inf)


def show_image_tensor(image_tensor):
    # 要求:使用交互式会话
    # 获取图像tensor对象对应的image对象,Image对象是一个[h,w,c]

    print(image_tensor)
    image = image_tensor.eval()
    # print(image)
    print('图像大小为:{}'.format(image.shape))

    if len(image.shape) == 3 and image.shape[2] == 1:
        # 黑白图像
        plt.imshow(image[:, :, 0], cmap='Greys_r')
        plt.show()
    elif len(image.shape) == 3:
        # 彩色图像
        plt.imshow(image)
        plt.show()


# 1. 交互式会话启动
session = tf.InteractiveSession()

image_path = './data/xiaoren.png'
# image_path = './data/gray.png'
# image_path = './data/black_white.jpg'

# 一,图像格式的转换
# 读取数据
file_contents = tf.read_file(image_path)

# 将图像数据转换为像素点的数据格式，返回对象为: [height, width, num_channels], 如果是gif的图像返回[num_frames, height, width, num_channels]
# height: 图片的高度的像素大小
# width: 图片的水平宽度的像素大小
# num_channels: 图像的通道数，也就是API中的channels的值
# num_frames: 因为gif的图像是一个动态图像，可以将每一个动的画面看成一个静态图像，num_frames相当于在这个gif图像中有多少个静态图像
# 参数channels：可选值：0 1 3 4，默认为0， 一般使用0 1 3，不建议使用4
# 0：使用图像的默认通道，也就是图像是几通道的就使用几通道
# 1：使用灰度级别的图像数据作为返回值（只有一个通道：黑白）
# 3：使用RGB三通道读取数据
# 4：使用RGBA四通道读取数据(R：红色，G：绿色，B：蓝色，A：透明度)

# 根据图片的格式有不同的decode api
# image_tensor = tf.image.decode_image(contents=file_contents, channels=3)
# image_tensor = tf.image.decode_jpeg(file_contents, channels=3)

image_tensor = tf.image.decode_png(contents=file_contents, channels=3)
# show_image_tensor(image_tensor)

# 二,图像大小重置

"""
BILINEAR = 0 线性插值，默认
NEAREST_NEIGHBOR = 1 最近邻插值，失真最小
BICUBIC = 2 三次插值
AREA = 3 面积插值
"""
# images: 给定需要进行大小转换的图像对应的tensor对象，格式为：[height, width, num_channels]或者[batch, height, width, num_channels]
# API返回值和images格式一样，唯一区别是height和width变化为给定的值
resize_image_tensor = tf.image.resize_images(images=image_tensor, size=(200, 200),
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# 展示一下
# show_image_tensor(resize_image_tensor)


# 三,图片的剪切和填充

# 图片重置大小,通过图片的剪切或者填充 (默认:从中间开始计算新图片的大小)
crop_or_pad_image_tensor = tf.image.resize_image_with_crop_or_pad(image_tensor, 200, 200)
# show_image_tensor(crop_or_pad_image_tensor)

# 中间等比例的剪切,central_fraction:0-1之间
central_crop_image_tensor = tf.image.central_crop(image_tensor, central_fraction=0.2)
# show_image_tensor(central_crop_image_tensor)

# 填充数据（给定位置开始填充）
pad_to_bounding_box_image_tensor = tf.image.pad_to_bounding_box(image_tensor, offset_height=400, offset_width=490,
                                                                target_height=1000,
                                                                target_width=1000)
# show_image_tensor(pad_to_bounding_box_image_tensor)

# 剪切数据（给定位置开始剪切）
crop_to_bounding_box_image_tensor = tf.image.crop_to_bounding_box(image_tensor, offset_height=10, offset_width=40,
                                                                  target_height=200, target_width=300)
# show_image_tensor(crop_to_bounding_box_image_tensor)

# 四,旋转
# 上下旋转
flip_up_down_image_tensor = tf.image.flip_up_down(image_tensor)
# show_image_tensor(flip_up_down_image_tensor)

# 水平旋转
flip_left_right_image_tensor = tf.image.flip_left_right(image_tensor)
# show_image_tensor(flip_left_right_image_tensor)

# 转置
transpose_image_tensor = tf.image.transpose_image(image_tensor)
# show_image_tensor(transpose_image_tensor)

# 旋转(k=1->90°,k=2->180°,k=3->270°,k=4->360°)
k_rot90_image_tensor = tf.image.rot90(image_tensor, k=3)
# show_image_tensor(k_rot90_image_tensor)

# 五,颜色空间的转换(RGB,HSV,GRAY)
# 颜色空间的转换必须将image的值转换为float32类型,不能是unit8类型
# 也就是都除以255
float32_image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)
# show_image_tensor(float32_image_tensor)

# rgb->hsv(h:图像的色度,s:图像的饱和度,v:图像的亮度)
hsv_image_tensor = tf.image.rgb_to_hsv(float32_image_tensor)
# show_image_tensor(hsv_image_tensor)

# hsv->rgb
rgb_image_tensor = tf.image.hsv_to_rgb(hsv_image_tensor)
# show_image_tensor(rgb_image_tensor)

# rgb->gray
gray_image_tensor = tf.image.rgb_to_grayscale(rgb_image_tensor)
# show_image_tensor(gray_image_tensor)

# 可以从颜色空间中提取图像的轮廓信息(图像的二值化)
a = gray_image_tensor
b = tf.less_equal(a, 0.9)
# 0是黑，1是白
# condition?true:false
# condition、x、y格式必须一模一样，当condition中的值为true的之后，返回x对应位置的值，否则返回y对应位置的值
# 对于a中所有大于0.9的像素值，设置为0
c = tf.where(condition=b, x=a, y=a - a)
# 对于a中所有小于等于0.9的像素值，设置为1
d = tf.where(condition=b, x=c - c + 1, y=c)
show_image_tensor(d)

# 六,图像的调整
