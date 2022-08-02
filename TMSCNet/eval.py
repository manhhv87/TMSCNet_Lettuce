import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from metrics.nmse import nmse
from metrics.nrmse import nrmse
from sklearn.metrics import r2_score
from data.data_preprocess import *
from data.read_label import *


#####################Parameter setting （参数设置）##########################
f='E:/dataset/lettuce/test/' #Original training data storage directory （原始训练数据存储目录）
read_json=f+'Images.json'  #Directory of file corresponding to test data （测试数据对应的文件的目录）
comp_json=f+'GroundTruth_All_388_Images.json' #Directory of label file corresponding to all data （全部数据对应的标签文件的目录）

box = [750, 250, 1350, 850]    #Crop the box of the original image （裁剪原图的方框）
shape=64 #Size of the picture after scaling （图片缩放后的尺寸）

model_save_path_11='h5/model_11.h5'  #Model file storage path （模型文件存储路径）
model_save_path_12='h5/model_12.h5'
model_save_path_13='h5/model_13.h5'
model_save_path_2='h5/model_2.h5'
model_save_path_3='h5/model_3.h5'

k_h=0.59 #Self correction coefficient （自矫正系数）
k_dia=0.98
k_fw=0.60
k_area=0.96

file_name='eval'  #Evaluating result storage file name （训练结果存储文件名称）
###########################################################################


#设置GPU的分配方式，以使网络正常运行
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)




##################数据处理
#Create folder （创建文件夹）
if os.path.isdir(f+'RGB_change')==False:
    os.mkdir(f+'RGB_change')
if os.path.isdir(f+'Debth_change')==False:
    os.mkdir(f+'Debth_change')


#Read in picture name （读入图片名）
name_rgb=[p.split('\\')[-1] for p in glob.glob(f+'RGB_*.png')]
name_debth=[p.split('\\')[-1] for p in glob.glob(f+'Debth_*.png')]


#Image cropping and zooming （图片裁剪及缩放）
if os.path.isfile(f+'RGB_change/'+name_rgb[0])==False:
    print('开始裁剪并缩放RGB图像！')
    print('Start cropping and scaling the RGB image!')
    [crop_resize(name,f,box,shape) for name in name_rgb]
    print('完成RGB图像的裁剪与缩放。')
    print('Finish cropping and scaling of RGB image.')

#Image cropping and zooming （图片裁剪及缩放）
if os.path.isfile(f+'Debth_change/'+name_debth[0])==False:
    print('开始裁剪并缩放深度图像！')
    print('Start cropping and scaling the depth image!')
    [debth_color(name,f,shape) for name in name_debth]
    print('完成深度图像的裁剪与缩放。')
    print('Finish cropping and scaling of depth image.')


#Read in label file （读入标签文件）
df1=pd.read_json(read_json)
df2=pd.read_json(comp_json)

df1=df1.drop(['ExperimentInfo', 'Varieties', 'Measurements', 'Camera'],axis=0)
df1=df1.drop(['General'],axis=1)
df2=df2.drop(['ExperimentInfo', 'Varieties', 'Measurements', 'Camera'],axis=0)
df2=df2.drop(['General'],axis=1)

value1=(df1.values).squeeze()
value2=(df2.values).squeeze()

rgb_name1=[v['RGBImage'] for v in value1]
rgb_name2=[v['RGB_Image'] for v in value2]
debth_name1=[v['DebthInformation'] for v in value1]
debth_name2=[v['Depth_Information'] for v in value2]

value=[]
for i in range(len(value1)):
    for j in range(len(value2)):
        if rgb_name1[i]==rgb_name2[j]:
            value.append(value2[j])

variety=[v['Variety'] for v in value]
fw=[v['FreshWeightShoot'] for v in value]
dw=[v['DryWeightShoot'] for v in value]
h=[v['Height'] for v in value]
dia=[v['Diameter'] for v in value]
area=[v['LeafArea'] for v in value]

kind=dict((num,var) for num,var in enumerate(np.unique(variety)))
variety=[list(kind.values()).index(var) for var in variety]



#Making datasets （制作数据集）
rgb_path=[]
debth_path=[]
for i in range(len(rgb_name1)):
    rgb_path.append(f+'RGB_change/'+rgb_name1[i])
    debth_path.append(f + 'Debth_change/' + debth_name1[i])

rgb_ds=[preprocess_val(p) for p in rgb_path]
debth_ds=[preprocess_val(p) for p in debth_path]

rgb_ds=np.array(rgb_ds)
debth_ds=np.array(debth_ds)



#Load model （加载模型）
model1=tf.keras.models.load_model(model_save_path_11)
model2=tf.keras.models.load_model(model_save_path_12)
model3=tf.keras.models.load_model(model_save_path_13)
model4=tf.keras.models.load_model(model_save_path_2)
model5=tf.keras.models.load_model(model_save_path_3)



#First stage model （一次模型）
[y_fw1, y_dw,y_h11,y_dia1, y_area1]=model1.predict(rgb_ds)
y_h12=model2.predict(debth_ds)
y_label=model3.predict(rgb_ds)
y_label=np.argmax(y_label,axis=1)  #取概率最大的索引作为类别


#Second stage model（二次模型）
x1=np.zeros((len(y_label),2))
for i in range(len(y_label)):
    x1[i,0]=y_dw[i]
    x1[i,1]=y_label[i]

x1=StandardScaler().fit_transform(x1)

[y_h2,y_dia2]=model4.predict(x1)

y_h=[]
[y_h.append(y_h12[i]*k_h+y_h2[i]*(1-k_h)) for i in range(len(y_h11))]
y_h=np.array(y_h)

y_dia=[]
[y_dia.append(y_dia1[i]*k_dia+y_dia2[i]*(1-k_dia)) for i in range(len(y_dia1))]
y_dia=np.array(y_dia)


#Third stage model（三次模型）
x2=np.zeros((len(y_label),4))
for i in range(len(y_label)):
    x2[i,0]=y_label[i]
    x2[i,1]=y_h[i]
    x2[i, 2] = y_dia[i]
    x2[i, 3] = y_dw[i]

x2=StandardScaler().fit_transform(x2)

[y_fw2,y_area2]=model5.predict(x2)

y_fw,y_area=[],[]
[y_fw.append(y_fw1[i]*k_fw+y_fw2[i]*(1-k_fw)) for i in range(len(y_fw1))]
[y_area.append(y_area1[i]*k_area+y_area2[i]*(1-k_area)) for i in range(len(y_area1))]

y_fw=np.array(y_fw)
y_area=np.array(y_area)


#Store results （存储结果）
if not os.path.exists('result/{}'.format(file_name)):
    os.makedirs('result/{}'.format(file_name))
np.save('result/{}/pred_fw'.format(file_name),y_fw)
np.save('result/{}/true_fw'.format(file_name),fw)
np.save('result/{}/pred_dw'.format(file_name),y_dw)
np.save('result/{}/true_dw'.format(file_name),dw)
np.save('result/{}/pred_h'.format(file_name),y_h)
np.save('result/{}/true_h'.format(file_name),h)
np.save('result/{}/pred_dia'.format(file_name),y_dia)
np.save('result/{}/true_dia'.format(file_name),dia)
np.save('result/{}/pred_area'.format(file_name),y_area)
np.save('result/{}/true_area'.format(file_name),area)


#r2、nrmse
r2_fw,nrmse_fw=r2_score(fw,y_fw),nrmse(fw,y_fw)
r2_dw,nrmse_dw=r2_score(dw,y_dw),nrmse(dw,y_dw)
r2_h,nrmse_h=r2_score(h,y_h),nrmse(h,y_h)
r2_dia,nrmse_dia=r2_score(dia,y_dia),nrmse(dia,y_dia)
r2_area,nrmse_area=r2_score(area,y_area),nrmse(area,y_area)

print(r2_fw,nrmse_fw)
print(r2_dw,nrmse_dw)
print(r2_h,nrmse_h)
print(r2_dia,nrmse_dia)
print(r2_area,nrmse_area)


#nmse
y_predict=np.zeros((len(value),5))
y_true=np.zeros((len(value),5))
for i in range(len(value)):
    y_predict[i,0]=y_fw[i]
    y_predict[i, 1] = y_dw[i]
    y_predict[i, 2] = y_h[i]
    y_predict[i, 3] = y_dia[i]
    y_predict[i, 4] = y_area[i]

    y_true[i,0]=fw[i]
    y_true[i, 1] = dw[i]
    y_true[i, 2] = h[i]
    y_true[i, 3] = dia[i]
    y_true[i, 4] = area[i]

eval=nmse(y_predict,y_true)
print(eval)


#Sort （排序）
y_fw=np.array(y_fw)[np.argsort(fw)]
fw=np.sort(fw)
y_dw=np.array(y_dw)[np.argsort(dw)]
dw=np.sort(dw)
y_h=np.array(y_h)[np.argsort(h)]
h=np.sort(h)
y_dia=np.array(y_dia)[np.argsort(dia)]
dia=np.sort(dia)
y_area=np.array(y_area)[np.argsort(area)]
area=np.sort(area)


#Plot results （绘制结果）
plt.subplots_adjust(wspace=0.2,hspace=0.5)  #调整子图间距

plt.subplot(3,2,1)
plt.title('fw')
plt.plot(np.arange(len(fw)),y_fw,c='r',label='Pred')
plt.plot(np.arange(len(fw)),fw,c='b',label='True')
plt.legend()

plt.subplot(3,2,2)
plt.title('dw')
plt.plot(np.arange(len(fw)),y_dw,c='r',label='Pred')
plt.plot(np.arange(len(fw)),dw,c='b',label='True')
plt.legend()

plt.subplot(3,2,3)
plt.title('h')
plt.plot(np.arange(len(fw)),y_h,c='r',label='Pred')
plt.plot(np.arange(len(fw)),h,c='b',label='True')
plt.legend()

plt.subplot(3,2,4)
plt.title('dia')
plt.plot(np.arange(len(fw)),y_dia,c='r',label='Pred')
plt.plot(np.arange(len(fw)),dia,c='b',label='True')
plt.legend()

plt.subplot(3,2,5)
plt.title('area')
plt.plot(np.arange(len(fw)),y_area,c='r',label='Pred')
plt.plot(np.arange(len(fw)),area,c='b',label='True')
plt.legend()

plt.show()
















