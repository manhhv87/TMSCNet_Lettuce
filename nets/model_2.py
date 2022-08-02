################加载库
import tensorflow as tf
from tensorflow.keras.layers import *



def Model_2(x):
    y = Flatten()(x)

    out_h = Dense(32, activation='relu')(y)
    out_h = Dense(64, activation='relu')(out_h)
    out_h = Dense(1, name='out_h')(out_h)

    out_dia = Dense(32, activation='relu')(y)
    out_dia = Dense(64, activation='relu')(out_dia)
    out_dia = Dense(1, name='out_dia')(out_dia)

    return [out_h,out_dia]



