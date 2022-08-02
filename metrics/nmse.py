import numpy as np

#y_pred=[ y_fw, y_dw, y_h, y_dia, y_area]
def nmse(y_predict,y_true):
    num=0
    den=0
    nmse=[]
    for j in range(5):
        num=0
        den=0
        for i in range(len(y_predict)):
            num+=np.square(y_true[i,j]-y_predict[i,j])  #分子
            den+=np.square(y_true[i,j]) #分母
        nmse.append(num/den)
    nmse=np.sum(nmse)

    return nmse

