'''
Created on Nov 19, 2012

@author: marcel
'''

import numpy as np
import scipy.io as sio
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def plot_mr(mag, dist, nmag, ndist, yfit, var):
    mag = np.reshape(mag, (ndist, nmag))
    dist = np.reshape(dist, (ndist, nmag))
    yfit = np.reshape(yfit, (ndist, nmag))
    var = np.reshape(var, (ndist, nmag))
    se = 2.0*np.sqrt(var)
    
    cmap = cm.get_cmap('jet')
    
    plt.subplots_adjust(hspace=0.5)
    
    plt.subplot(121)
    plt.contourf(mag,dist,yfit,25,cmap=cmap)
    #plt.contourf(x,y,yfit,25,cmap=cmap)
    plt.colorbar()
    plt.subplot(122)
    plt.contourf(mag,dist,se,25,cmap=cmap)
    #plt.contourf(x,y,se,25,cmap=cmap)
    plt.colorbar()

    plt.xlabel('Mag')
    plt.ylabel('Dist')
    
    plt.show()

def load_test_data(fname):
    mat_dict = sio.loadmat(fname)
    X = mat_dict['X']
    return X


def load_pred(fname):
    Ypred = np.loadtxt(fname, delimiter=',')
    yfit  = Ypred[:,0]
    var = Ypred[:,1]
    
    return yfit, var

if __name__ == '__main__':
    test_fname = '/home/marcel/datasets/multilevel/eusinan/bssa/viz/viz_eudata_test.mat'
    pred_fname = '/home/marcel/datasets/multilevel/eusinan/bssa/viz/pga/gpardse_corrnoise_pred.csv'
    
    mag_idx = 0
    dist_idx = 5
   
    X = load_test_data(test_fname)
    yfit, var = load_pred(pred_fname)
    
    rec_mask_rev = np.all(np.c_[X[:,1] == 5,X[:,6]==760,X[:,2]==1],1)
    rec_mask_ss = np.all(np.c_[X[:,1] == 5,X[:,6]==760,X[:,3]==1],1)
    rec_mask_normal = np.all(np.c_[X[:,1] ==5,X[:,6]==760,X[:,4]==1],1)
    
    nmag = len(np.unique1d(X[:,mag_idx]))
    ndist = len(np.unique1d(X[:,dist_idx]))
    
    
    mag = X[rec_mask_rev,mag_idx]
    dist = X[rec_mask_rev,dist_idx]
    
    
    plot_mr(mag,dist,nmag,ndist,yfit[rec_mask_rev],var[rec_mask_rev])
    plot_mr(mag,dist,nmag,ndist,yfit[rec_mask_ss],var[rec_mask_ss])
    plot_mr(mag,dist,nmag,ndist,yfit[rec_mask_normal],var[rec_mask_normal])
    
    
    
    
    
    
    