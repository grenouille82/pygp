'''
Created on Nov 20, 2012

@author: marcel
'''

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import rc


def plot_groupbarerror(means, errors):
    '''
    '''
    width = 0.1
    n,m = means.shape
    
    
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    
    for i in xrange(n):   
        print means[i]
        ax.bar(np.arange(m)+i*width, means[i], width, color='g', yerr=errors[i])
    
    plt.show()

def load_eval_results(fname):
    #dt = ['s100']
    types = ['S100']
    types.extend([np.float]*18)
    #types = list(types)
    names = ['method']
    names.extend([str(i) for i in xrange(18)])
    #names = list(names)
    dt = np.dtype(zip(names, types))
    
    data = np.loadtxt(fname, delimiter=',', skiprows=2, dtype=dt, comments='#')
    
    methods = list(data['method'])
    values = data[[str(i) for i in xrange(18)]]
    values = values.view(np.float).reshape(values.shape + (-1,))
    means = values[:,np.arange(0,18,3)]
    stddev = values[:,np.arange(1,18,3)]
    se = values[:,np.arange(2,18,3)]
    return methods, means, stddev, se
    
if __name__ == '__main__':
    fname = '/home/marcel/datasets/multilevel/eusinan/bssa/results/cv_gpmodels_splitz2_nmll.csv'
    methods, means, stddev, se = load_eval_results(fname)
    
    plot_groupbarerror(means, stddev)
    