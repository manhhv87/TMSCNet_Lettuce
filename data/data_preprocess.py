import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import cv2


def crop_resize(name,f,box,shape):
    img=Image.open(f+name)
    img=img.crop(box)
    img=img.resize((shape,shape))
    img.save(f+'RGB_change/'+name)


def debth_color(name,f,shape):
    img = (cv2.imread(f + name))[300:800, 800:1300]  # 裁剪
    img=cv2.resize(img,(shape,shape))
    img=cv2.applyColorMap(cv2.convertScaleAbs(img,alpha=30),cv2.COLORMAP_JET)
    img=Image.fromarray(img)
    img.save(f+'Debth_change/'+name)

def augment(name,loop_num,f):
    #创建图片增强对象
    datagen = ImageDataGenerator(
        rotation_range=5*np.random.rand(),
        width_shift_range=0.01,
        height_shift_range=0.01,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    #图片增强
        #读取及准备
    for i in range(len(name)):
        img = load_img(f + 'RGB_change/' + name[i])  #加载图像
        img = img_to_array(img)  # 图像转换成数组
        img = img.reshape((1,) + img.shape)  # 扩充维度，满足数据生成要求

        #图片扩充
        k, j = 0, 0
        for batch in datagen.flow(img, batch_size=1,
                                  save_to_dir=f+'RGB_aug',
                                  save_prefix=os.path.splitext(name[i])[0],
                                  save_format='png',
                                  seed=1):  # 设置随机种子，每次生成的图的效果相同

            k += 1
            if k >loop_num:
                break


def augment_depth(name,loop_num,f):
    #创建图片增强对象
    datagen = ImageDataGenerator(
        rotation_range=5*np.random.rand(),
        width_shift_range=0.01,
        height_shift_range=0.01,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    #图片增强
        #读取及准备
    for i in range(len(name)):
        img = load_img(f + 'Debth_change/' + name[i])  #加载图像
        img = img_to_array(img)  # 图像转换成数组
        img = img.reshape((1,) + img.shape)  # 扩充维度，满足数据生成要求

        #图片扩充
        k, j = 0, 0
        for batch in datagen.flow(img, batch_size=1,
                                  save_to_dir=f+'Debth_aug',
                                  save_prefix=os.path.splitext(name[i])[0],
                                  save_format='png',
                                  seed=1):  # 设置随机种子，每次生成的图的效果相同

            k += 1
            if k >loop_num:
                break


def preprocess(path):
    #读入
    img=tf.io.read_file(path)
    img=tf.image.decode_png(img,channels=3)
    #归一化
    img=tf.cast(img,tf.float32)
    img=img/255
    return img


def preprocess_val(path):
    img=Image.open(path)
    img=np.array(img)
    img=img/255.
    return img
