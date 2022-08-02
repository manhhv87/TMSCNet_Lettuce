################加载库
import tensorflow as tf
from tensorflow.keras.layers import *



def Model_3(x):
    y = Flatten()(x)

    out_fw = Dense(32, activation='relu')(y)
    out_fw = Dense(64, activation='relu')(out_fw)
    out_fw = Dense(1, name='out_fw')(out_fw)

    out_area = Dense(32, activation='relu')(y)
    out_area = Dense(64, activation='relu')(out_area)
    out_area = Dense(1, name='out_area')(out_area)

    return [out_fw,out_area]



