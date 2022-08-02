import numpy as np

def nrmse(y_true,y_predict):
    num=0
    den=0
    for i in range(len(y_predict)):
        num+=np.square(y_true[i]-y_predict[i])  #分子
        den+=np.square(y_true[i]) #分母
    return (np.sqrt(num)/np.sqrt(den))[0]*100
