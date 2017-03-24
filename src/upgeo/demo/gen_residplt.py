'''
Created on Nov 17, 2012

@author: marcel
'''
import os

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import rc
from upgeo.demo.util import loadmat_data


def plot_resid(pred, actual, x, xlabel=None, xticks=None, var=None, fname=None):
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    #matplotlib.rc('font', **font)
    
    rc('text', usetex=True)
    rc('font', family='serif')
    
    resid = pred-actual
    plt.figure(figsize=(16,8))
    plt.ylim(-5,5)
    plt.plot(x, resid, 'wo') 
    
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(r"residual", fontsize=24)
    #plt.xlabel(r"\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!", fontsize=20)
    if xticks == None:
        plt.xticks(fontsize=20)
    else:
        plt.xticks(xticks, fontsize=20)
    plt.yticks(np.arange(-5,5.5,0.5), fontsize=20)
    plt.grid(True)  
    #plt.xscale('symlog')
    xmin, xmax = plt.xlim()
    plt.plot([xmin,xmax],[0,0], color='k', linestyle='-', linewidth=2)
    if fname == None:
        plt.show()
    else:
        plt.savefig(fname + ".eps")
        plt.savefig(fname + ".png")
    
    plt.clf()
    
def ensure_dir(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)

   
if __name__ == '__main__':
    datafile = '/home/marcel/datasets/multilevel/eusinan/bssa/eval_eudata.mat'
    evalpath = '/home/marcel/datasets/multilevel/eusinan/bssa/results/T1/splitz1'  
    
    
    X,Y = loadmat_data(datafile)
    
    actual = Y[:,6]
    mean = np.mean(actual)
    
    ensure_dir(os.path.join(evalpath, 'plots'))
    for file in os.listdir(evalpath):
        if file != 'plots':
            print 'file: {0}'.format(file)
            result = np.loadtxt(os.path.join(evalpath, file), delimiter=',')
            if not 'testerror' in file:
                pred = result[:,0] + mean
            plot_resid(pred, actual, X[:,6], xlabel=r"$\displaystyle R_{JB}$ (km)", xticks=[0, 25, 50, 75, 100, 125, 150,175, 200], 
                       fname=os.path.join(evalpath,'plots',file+'_resid_dist'))
            
            plot_resid(pred, actual, X[:,0], xlabel=r"$\displaystyle M$",
                       fname=os.path.join(evalpath,'plots',file+'_resid_mag'))
            
            plot_resid(pred, actual, X[:,2], xlabel=r"Focal Depth", xticks=[0,5,10,15,20,25,50,75,100],
                       fname=os.path.join(evalpath,'plots',file+'_resid_depth'))
            
            plot_resid(pred, actual, X[:,9], xlabel=r"$\displaystyle Vs_{30}$", xticks=[0,250,500,750,1000,1500,2000,2500],
                       fname=os.path.join(evalpath,'plots',file+'_resid_vs30'))
    
