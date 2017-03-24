'''
Created on Nov 13, 2012

@author: marcel
'''

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
from upgeo.base.kernel import SEKernel, LinearKernel
from scipy.stats import norm
from itertools import cycle
from upgeo.base.kernel import SEKernel

if __name__ == '__main__':
    rc('text', usetex=True)
    rc('font', family='serif')
    
    kernel = SEKernel(np.log(1), np.log(1))
    xmin = -10
    xmax = 10
    step = 0.05
    n = 3

    lines = ["k-","k--","k-.","k:","k.",  "k:.","k:.-"]
    linecycler = cycle(lines)
    
    
    x = np.arange(xmin, xmax, step)
    x = np.r_[x, xmax]
    x = x[:,np.newaxis]
    m = len(x)
    
    K = kernel(x)
    L = np.linalg.cholesky(K+np.eye(m)*1e-9)
    
    Y = np.zeros((m,n))
    
    for i in xrange(n):
        u = np.random.randn(m)
        y = np.dot(L, u)
        plt.plot(x,y)
        #plt.plot(x,y,next(linecycler))
        Y[:,i] = y
    
    
    
    mean = 0
    var = np.diag(K)
    sd = np.sqrt(var)
    
    f = np.r_[mean+2.0*sd, (mean-2.0*sd)[::-1]]
    plt.fill(np.r_[x, x[::-1]], f, edgecolor='w', facecolor='#d3d3d3')
    
    plt.xlabel(r"input, $\displaystyle x$", fontsize=24)
    plt.ylabel(r"output, $\displaystyle f(x)$", fontsize=24)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    print xmin
    print xmax
    plt.xlim(xmin,xmax)
    plt.ylim(-3,3)
    
    plt.show()
    
    np.savetxt('/home/marcel/gp_prior_samples.csv', np.c_[x,Y], delimiter=',')
    np.savetxt('/home/marcel/gp_prior_confidence.csv', np.c_[x,mean+2.0*sd,mean-2.0*sd], delimiter=',')
    