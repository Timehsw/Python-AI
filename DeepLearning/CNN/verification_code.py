# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/22
    Desc : 验证码识别
    Note : 假定验证码中只有数字和大小写字母eg:Gx3f
'''

import random
import numpy as np
# captcha是python验证码的库，安装方式pip install captcha
from captcha.image import ImageCaptcha
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

'''
1. 使用训练集进行网络训练
训练集数据怎么来？？===> 使用代码随机的生成一批验证码数据（最好不要每次都随机一个验证码），最好是先随机出10w张验证码的图片，然后利用这10w张图片来训练；否则收敛会特别慢，而且有可能不收敛
如何训练？直接将验证码输入（输入Gx3f），神经网络的最后一层是4个节点，每个节点输出对应位置的值(第一个节点输出：G，第二个节点输出：x，第三个节点输出：3，第四个节点输出：f)
2. 使用测试集对训练好的网络进行测试
3. 当测试的正确率大于75%的时候，模型保存
4. 加载模型，对验证码进行识别
'''
code_char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 'q', 'a', 'z', 'w', 's', 'x', 'e', 'd', 'c', 'r',
                 'f', 'v', 't', 'g', 'b', 'y', 'h', 'n', 'u', 'j',
                 'm', 'i', 'k', 'o', 'l', 'p', 'Q', 'A', 'Z', 'W',
                 'S', 'X', 'E', 'D', 'C', 'R', 'F', 'V', 'T', 'G',
                 'B', 'Y', 'H', 'N', 'U', 'J', 'M', 'I', 'K', 'O',
                 'L', 'P']


def random_code_text(code_size=4):
    '''
    随机产生验证码字符
    :param code_size:
    :return:
    '''
    code_text = []
    for i in range(code_size):
        c = random.choice(code_char_set)
        code_text.append(c)
    return code_text


def generate_code_image(code_size=4):
    '''
    随机产生一个验证码图像的对象
    :param code_size:
    :return:
    '''
    image = ImageCaptcha()
    code_text = random_code_text(code_size)
    code_text = ''.join(code_text)
    # 将字符串转换为验证码
    captcha = image.generate(code_text)
    # 如果要保存验证码
    image.write(code_text, 'data/captcha/' + code_text + ".jpg")
    print(captcha)

    # 将验证码转换为图片的形式
    code_image = Image.open(captcha)
    code_image = np.array(code_image)
    return code_text, code_image


def show_code_image(text, image):
    ax = plt.figure()
    ax.text(0.1, 0.9, text, ha='center', va='center')
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    text, image = generate_code_image()
    show_code_image(text, image)
