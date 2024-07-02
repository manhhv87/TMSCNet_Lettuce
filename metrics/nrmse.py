import numpy as np


def nrmse(y_true, y_predict):
    num = 0
    den = 0
    for i in range(len(y_predict)):
        num += np.square(y_true[i] - y_predict[i])  # numerator
        den += np.square(y_true[i])  # denominator
    return (np.sqrt(num)/np.sqrt(den))[0]*100
