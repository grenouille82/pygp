'''
Created on Oct 1, 2012

@author: marcel
'''

import numpy as np
import scipy.io as sio

def loadmat_folds(filename):
    '''
    Load the training and test folds from data. The method
    returns the indices of the training and test samples.
    '''
    mat_dict = sio.loadmat(filename)
    train = np.squeeze(mat_dict['tr'])
    test = np.squeeze(mat_dict['tst'])
    
    #
    train = train-1
    test = test-1
    return (train, test)

def loadmat_data(filename):
    mat_dict = sio.loadmat(filename)
    X = np.squeeze(mat_dict['X'])
    #Y = np.squeeze(mat_dict['Y'])
    Y = np.squeeze(mat_dict['y'])
    return (X,Y)

def loadmat_data_and_periods(filename):
    mat_dict = sio.loadmat(filename)
    X = np.squeeze(mat_dict['X'])
    Y = np.squeeze(mat_dict['Y'])
    periods = np.squeeze(mat_dict['periods'])
    return (X,Y,periods)

def loadmat_transfer_data(filename):
    mat_dict = sio.loadmat(filename)
    Xt = np.squeeze(mat_dict['Xt'])
    yt = np.squeeze(mat_dict['yt'])
    Xb = np.squeeze(mat_dict['Xb'])
    yb = np.squeeze(mat_dict['yb'])
    return Xt,yt,Xb,yb

def loadmat_mtl_data(filename):
    mat_dict = sio.loadmat(filename)
    X = np.squeeze(mat_dict['X'])
    y = np.squeeze(mat_dict['y'])
    tasks = np.squeeze(mat_dict['task'])
    return X,y,tasks

def prepare_mtl_data(X,Y):
    '''
    @todo: - handling missing values in Y
    '''
    Xmtl = np.empty((0,X.shape[1]))
    Ymtl = np.empty((0))
    ntasks = Y.shape[1]
    itasks = np.zeros(ntasks)
    
    for i in xrange(ntasks):
        idx = ~np.isnan(Y[:,i])
        #print 'idx={0}'.format(idx)
        Xmtl = np.r_[Xmtl, X[idx]]
        Ymtl = np.r_[Ymtl, Y[idx,i]]
        if i != ntasks-1:
            itasks[i+1] = itasks[i] + np.sum(idx)
    
    return Xmtl, Ymtl, itasks
    
    
    
    
