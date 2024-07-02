import tensorflow as tf
from tensorflow.keras.layers import *


def Model_13(x, n_filters, layers, kernel_size, pool_size, dropout, units, n):
    filters = n_filters
    for i in range(layers):
        x = Conv2D(filters=filters,
                   kernel_size=(kernel_size, kernel_size),
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(pool_size, pool_size), strides=2)(x)
        x = Dropout(dropout)(x)
        filters *= 2

    x = Flatten()(x)

    x = Dense(units, activation='relu')(x)
    x = Dense(units / n, activation='relu')(x)
    x = Dense(units / 2 * n, activation='relu')(x)
    x = Dense(units / 4 * n, activation='relu')(x)
    x = Dense(units / 8 * n, activation='relu')(x)
    out_h = Dense(1, activation='linear', name='out_h')(x)

    return out_h
