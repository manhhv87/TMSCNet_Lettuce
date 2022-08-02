import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array


def load_json(f_json):
    #读入
    df=pd.read_json(f_json)
    #删除多余行
    df=df.drop(['ExperimentInfo', 'Varieties', 'Measurements', 'Camera'],axis=0)
    df=df.drop(['General'],axis=1)

    #获取值和索引
    index=[i.split('Image')[-1] for i in list(df.index)]  #图片顺序
    value=(df.values).squeeze()

    #分离各个变量
    variety=[v['Variety'] for v in value]
    fw=[v['FreshWeightShoot'] for v in value]
    dw=[v['DryWeightShoot'] for v in value]
    h=[v['Height'] for v in value]
    dia=[v['Diameter'] for v in value]
    area=[v['LeafArea'] for v in value]

    #把类别转换为数字
    kind=dict((num,var) for num,var in enumerate(np.unique(variety)))
    variety=[list(kind.values()).index(var) for var in variety]

    return index,fw,dw,h,dia,area,variety