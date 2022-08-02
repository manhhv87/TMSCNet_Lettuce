import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from data.data_preprocess import *
from data.read_label import *
from sklearn.preprocessing import StandardScaler
from nets.model_2 import *



#####################Parameter setting （参数设置）##########################
f='E:/dataset/lettuce/train/'   #Original training data storage directory （原始训练数据存储目录）
f_json=f+'GroundTruth_SendJuly13.json'  #Directory of label file corresponding to training data （训练数据对应的标签文件的目录）

batch_size=16  #Number of batches of images loaded （加载图像的批次数）
validation_split = 0.2 #Partition ratio of training set and verification set （训练集和验证集的划分比率）

learning_rate=0.001  #Learning rate （学习率）
epochs=200  #Number of cycles （循环次数）

model_save_path='h5/model_2.h5'  #Model file storage path （模型文件存储路径）
###########################################################################

#Read in label file （读入标签文件）
index,fw,dw,h,dia,area,variety=load_json(f_json)
var_ds,dia_ds,h_ds,dw_ds,fw_ds,area_ds=np.array(variety),np.array(dia),np.array(h),np.array(dw),np.array(fw),np.array(area)

#Making datasets （制作数据集）
x=np.zeros((len(variety),2))
y_h=np.zeros(len(variety))
y_dia=np.zeros(len(variety))

for i in range(len(variety)):
    x[i, 0] = dw_ds[i]
    x[i,1]=var_ds[i]
    y_h[i]=h_ds[i]
    y_dia[i]=dia_ds[i]

x=StandardScaler().fit_transform(x)


#Pairing out of order （配对打乱顺序）
np.random.seed(116)
np.random.shuffle(x)
np.random.seed(116)
np.random.shuffle(y_h)
np.random.seed(116)
np.random.shuffle(y_dia)



#Establish network （建立网络）
inputs=Input(shape=(2,))
model=tf.keras.Model(inputs=inputs,outputs=Model_2(inputs))


#Compile model （编译模型）
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse',
)

#Training model （训练模型）
model.fit(x,{'out_h':y_h,'out_dia':y_dia},
              shuffle=True,
              epochs=epochs,batch_size=batch_size,
              validation_split=validation_split)

#Save model （保存模型）
model.save(model_save_path)

#Print model （打印模型）
model.summary()



