'''
Created on Dec 1, 2012

@author: marcel
'''
import numpy as np 

def load_train_data(fname, pred_cols, target_cols):
    data = np.loadtxt(fname, delimiter=',', skiprows=1)
    X = data[:, pred_cols]
    Y = data[:, target_cols]
    
    return X,Y