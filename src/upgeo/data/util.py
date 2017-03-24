'''
Created on Sep 5, 2012

@author: marcel
'''
import scipy.io as sio


def loadmat_regdata(filename):
    mat_dict = sio.loadmat(filename)
    X = mat_dict['X']
    y = mat_dict['y']
    return X,y
