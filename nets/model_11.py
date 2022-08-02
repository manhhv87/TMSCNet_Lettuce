################加载库
import tensorflow as tf
from tensorflow.keras.layers import *





def Model_11(x,n_filters,layers,kernel_size,pool_size,dropout,units,n):
    filters=n_filters
    for i in range(layers):
        x = Conv2D(filters=filters,
                   kernel_size=(kernel_size, kernel_size),
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(pool_size, pool_size), strides=2)(x)
        x = Dropout(dropout)(x)
        filters *= 2

    x=Flatten()(x)

    x1 = Dense(units, activation='relu')(x)
    x1 = Dense(units / n, activation='relu')(x1)
    x1 = Dense(units / 2*n, activation='relu')(x1)
    x1 = Dense(units / 4*n, activation='relu')(x1)
    x1 = Dense(units / 8 * n, activation='relu')(x1)
    out_fw = Dense(1,activation='linear', name='out_fw')(x1)

    x2 = Dense(units, activation='relu')(x)
    x2 = Dense(units / n, activation='relu')(x2)
    x2 = Dense(units / 2*n, activation='relu')(x2)
    x2 = Dense(units / 4*n, activation='relu')(x2)
    x2 = Dense(units / 8 * n, activation='relu')(x2)
    out_dw = Dense(1,activation='linear',  name='out_dw')(x2)

    x3 = Dense(units, activation='relu')(x)
    x3 = Dense(units / n, activation='relu')(x3)
    x3 = Dense(units / 2*n, activation='relu')(x3)
    x3 = Dense(units / 4*n, activation='relu')(x3)
    x3 = Dense(units / 8 * n, activation='relu')(x3)
    out_h = Dense(1, activation='linear', name='out_h')(x3)

    x4 = Dense(units, activation='relu')(x)
    x4 = Dense(units / n, activation='relu')(x4)
    x4 = Dense(units / 2*n, activation='relu')(x4)
    x4 = Dense(units / 4*n, activation='relu')(x4)
    x4 = Dense(units / 8 * n, activation='relu')(x4)
    out_dia = Dense(1,activation='linear',  name='out_dia')(x4)

    x5 = Dense(units, activation='relu')(x)
    x5 = Dense(units / n, activation='relu')(x5)
    x5 = Dense(units / 2*n, activation='relu')(x5)
    x5 = Dense(units / 4*n, activation='relu')(x5)
    x5 = Dense(units / 8 * n, activation='relu')(x5)
    out_area = Dense(1,activation='linear',  name='out_area')(x5)

    return [out_fw,out_dw,out_h,out_dia,out_area]



