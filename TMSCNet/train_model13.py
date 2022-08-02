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
from nets.model_13 import *
import cv2



#####################Parameter setting （参数设置）##########################
f='E:/dataset/lettuce/train/'   #Original training data storage directory （原始训练数据存储目录）
f_json=f+'GroundTruth_SendJuly13.json'  #Directory of label file corresponding to training data （训练数据对应的标签文件的目录）

shape=64 #Size of the picture after scaling （图片缩放后的尺寸）
loop_num=100 #Multiple of image enhancement （图像增强的倍数）

batch_size=500  #Number of batches of images loaded （加载图像的批次数）

units=128  #Number of neurons in FCN （FCN的神经元个数）
kernel_size=3  #Convolution kernel size （卷积核尺寸）
pool_size=2  #Pool core size （池化核尺寸）
dropout=0.1  #Dropout ratio （丢弃比率）
n_filters=32 #Number of convolution kernels （卷积核个数）
layers=1  #Layers of convolution kernel （卷积核的层数）
n=4  #Reduction times of neurons （神经元的缩小倍数）

learning_rate=0.001  #Learning rate （学习率）
epochs=50  #Number of cycles （循环次数）

model_save_path='h5/model_13.h5'  #Model file storage path （模型文件存储路径）
file_name='Model_13'  #Training result storage file name （训练结果存储文件名称）
###########################################################################



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#Create folder （创建文件夹）
if os.path.isdir(f+'Debth_change')==False:
    os.mkdir(f+'Debth_change')
if os.path.isdir(f+'Debth_aug')==False:
    os.mkdir(f+'Debth_aug')

#Read in picture name （读入图片名）
name=[p.split('\\')[-1] for p in glob.glob(f+'Debth_*.png')]

#Image cropping and zooming （图片裁剪及缩放）
if os.path.isfile(f+'Debth_change/'+name[0])==False:
    print('开始裁剪并缩放图像！')
    print('Start cropping and scaling the image!')
    [debth_color(name,f,shape) for name in name]
    print('完成裁剪与缩放。')
    print('Finish cropping and scaling.')


#Image enhancement （图像增强）
if os.path.isfile(f+'Debth_aug/'+name[0].split('.')[0]+'_0_1026.png')==False:
    print('开始执行数据增强！')
    print('Start data enhancement!')
    augment_depth(name,loop_num,f)
    print('完成数据增强。')
    print('Complete data enhancement.')


#Read in label file （读入标签文件）
index,fw,dw,h,dia,area,variety=load_json(f_json)


#Get the enhanced image path （获取增强后的图像路径）
aug_path=glob.glob(f+'Debth_aug/*.png')[:loop_num]
aug_name=[p.split('\\')[-1].split('Debth_100')[-1] for p in aug_path]

img_path=[]
for i in range(len(index)):
    for j in range(loop_num):
        img_path.append(f+'Debth_aug/Debth_'+index[i]+aug_name[j])

#Label repeat loop_ Num times （标签重复loop_num次）
h_ds=[]
for i in range(len(index)):
    for j in range(loop_num):
        h_ds.append(h[i])
h_ds=np.array(h_ds)


#Pairing out of order （配对打乱顺序）
np.random.seed(116)
np.random.shuffle(img_path)
np.random.seed(116)
np.random.shuffle(h_ds)


#Making datasets （制作数据集）
img_ds=tf.data.Dataset.from_tensor_slices(img_path).map(preprocess)
h_ds=tf.data.Dataset.from_tensor_slices(h_ds)

dataset=tf.data.Dataset.zip((img_ds,h_ds))
dataset=dataset.repeat().shuffle(100).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

n_test=int(len(img_path)*0.2)
n_train=len(img_path)-n_test
dataset_train=dataset.skip(n_test)
dataset_test=dataset.take(n_train)


#Establish network （建立网络）
inputs=Input(shape=(shape,shape,3))
outputs=Model_13(inputs,n_filters,layers,kernel_size,pool_size,dropout,units,n)
model=tf.keras.Model(inputs=inputs,outputs=outputs)


#Compile model （编译模型）
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse',
)

#Training model （训练模型）
print('训练开始前有较长的数据处理时间，请耐心等待！')
print('There is a long data processing time before the training starts, please wait patiently!')
history=model.fit(dataset_train,
                  steps_per_epoch=n_train//batch_size,
                  epochs=epochs,
                  validation_data=dataset_test,
                  validation_steps=n_test//batch_size,
                  shuffle=True
                  )

#Save model （保存模型）
model.save(model_save_path)

#Print model （打印模型）
model.summary()


#Store training parameters （存储训练参数）
if not os.path.exists('result/{}'.format(file_name)):
    os.makedirs('result/{}'.format(file_name))
np.save('result/{}/train_loss_h.npy'.format(file_name),history.history['loss'])
np.save('result/{}/test_loss_h.npy'.format(file_name),history.history['val_loss'])









