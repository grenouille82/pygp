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
from upgeo.base.gp import GPRegression

if __name__ == '__main__':
    rc('text', usetex=True)
    rc('font', family='serif')
    
    kernel = SEKernel(np.log(1), np.log(1))
    gp = GPRegression(kernel)
    
    xmin = -10
    xmax = 10
    step = 0.05
    n = 3
    k = 5
    
    Xtrain = np.random.uniform(xmin, xmax, k)
    Xtrain = Xtrain[:,np.newaxis]
    ytrain = np.random.uniform(-3, 3, k)
    gp.fit(Xtrain, ytrain)

    lines = ["k-","k--","k-.","k:","k.",  "k:.","k:.-"]
    linecycler = cycle(lines)
    
    
    x = np.arange(xmin, xmax, step)
    x = np.r_[x, xmax]
    x = x[:,np.newaxis]
    m = len(x)
    
    mean, sigma = gp.posterior(x)
    L = np.linalg.cholesky(sigma+np.eye(m)*1e-9)
    Y = np.zeros((m,n))
    
    for i in xrange(n):
        u = np.random.randn(m)
        y = mean + np.dot(L, u)
        plt.plot(x,y)
        #plt.plot(x,y,next(linecycler))
        Y[:,i] = y
    
    yfit, var = gp.predict(x, ret_var=True)
    sd = np.sqrt(var)
    
    f = np.r_[yfit+2.0*sd, (yfit-2.0*sd)[::-1]]
    plt.fill(np.r_[x, x[::-1]], f, edgecolor='w', facecolor='#d3d3d3')
    
    plt.plot(Xtrain, ytrain, '+', markersize=18, markerfacecolor='black', markeredgecolor='black', markeredgewidth=2)
    
    plt.xlabel(r"input, $\displaystyle x$", fontsize=24)
    plt.ylabel(r"output, $\displaystyle f(x)$", fontsize=24)
    
    np.savetxt('/home/marcel/gp_posterior_samples5.csv', np.c_[x,Y], delimiter=',')
    np.savetxt('/home/marcel/gp_posterior_traindata5.csv', np.c_[Xtrain,ytrain], delimiter=',')
    np.savetxt('/home/marcel/gp_posterior_confidence5.csv', np.c_[x,yfit+2.0*sd,yfit-2.0*sd], delimiter=',')
    
    plt.xlim(xmin,xmax)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
