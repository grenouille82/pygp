'''
Created on Aug 8, 2011

@author: marcel
'''
import scipy
import numpy as np
import pylab as pl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from upgeo.base.kernel import SEKernel, LinearKernel
from scipy.stats import norm
from itertools import cycle
from matplotlib import rc

def plot1d_kernel(kernel, xmin, xmax, step, n=5):
    rc('text', usetex=True)
    rc('font', family='serif')
    if xmin >= xmax:
        raise ValueError('xmin must be smaller than xmax.')
    
    lines = ["k-","k--","k-.","k:","k.",  "k:.","k:.-"]
    linecycler = cycle(lines)
    
    
    x = np.arange(xmin, xmax, step)
    x = np.r_[x, xmax]
    x = x[:,np.newaxis]
    m = len(x)
    
    K = kernel(x)
    K = K + np.eye(m)*1e-9
    L = np.linalg.cholesky(K)
    Y = np.zeros((m,n))
    for i in xrange(n):
        u = np.random.randn(m)
        y = np.dot(L, u)
        plt.plot(x,y)
        Y[:,i] = y
        #plt.plot(x,y,next(linecycler))
    
    plt.xlabel(r"input, $\displaystyle x$", fontsize=24)
    plt.ylabel(r"output, $\displaystyle f(x)$", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-3,3)
    plt.show()
    
    np.savetxt('/home/marcel/gp_sekernel_l0.05_samples.csv', np.c_[x,Y], delimiter=',')
    

def plot1d_gp(gp, xmin, xmax, n=100):
    '''
    validate 1 dimensionality
    '''
    if xmin >= xmax:
        raise ValueError('xmin must be smaller than xmax.')
    
    #test cases
    data = gp.training_set
    likel = gp.log_likel
    params = gp.hyperparams
    
    x = np.linspace(xmin, xmax, num=n)
    x = x[:,np.newaxis]
    mean, var = gp.predict(x, ret_var=True)
    mean = np.squeeze(mean)
    sd = np.sqrt(var)
    
    
    pl.figure()
    
    #plot mean and std error area
    pl.plot(x, mean, color='k', linestyle=':')
    f = np.r_[mean+2.0*sd, (mean-2.0*sd)[::-1]]
    pl.fill(np.r_[x, x[::-1]], f, edgecolor='w', facecolor='#d3d3d3')
    
    #plot training samples
    pl.plot(data[0], data[1], 'rs')
    
    #print plot title
    t = 'Log likelihood: {0}\n{1}'.format(likel, np.exp(params))
    pl.title(t)
    pl.show()
    
def plot1d_cov_gp(gp, xmin, xmax, n=100):
    
    likel = gp.log_likel
    params = gp.hyperparams
    kernel = gp.kernel
    
    x = np.linspace(xmin, xmax, num=n)
    x = x[:,np.newaxis]
    
    cov = kernel(x)    
    
    
    cmap = cm.get_cmap('jet') 
    plt.imshow(cov, interpolation='nearest', cmap=cmap)
    t = 'Log likelihood: {0}\n{1}'.format(likel, np.exp(params))
    plt.title(t)
    plt.colorbar()
    plt.show()

def plot_gp_prior(kernel, xmin, xmax, n=10):
    
    for i in xrange(n):
        x,y = gendata_gp_1d(kernel, xmin, xmax, 100, 0)
        plt.plot(x,y)
        
    plt.show()
    
def plot_cov(kernel, xmin, xmax, n=100):
    x = np.linspace(xmin, xmax, num=n)
    x = x[:,np.newaxis]
    
    cov = kernel(x)
    cmap = cm.get_cmap('jet') 
    plt.imshow(cov, interpolation='nearest', cmap=cmap, vmin=0, vmax=1, extent=(-1,1,1,-1))
    plt.colorbar()
    plt.show()
    
    

def gendata_gp_1d(kernel, min, max, n=100, sigma=1, seed=None):
    '''
    '''
    if min > max:
        raise ValueError('min must be less than max')
    if sigma < 0:
        raise ValueError('sigma must be >= 0.')
    
    if seed != None:
        np.random.seed(seed)

    
    x = np.linspace(min, max, n)
    x = x[:,np.newaxis]
    
    K = kernel(x)
    y = np.random.multivariate_normal(np.zeros(n), K)
    if sigma > 0:
        y += np.random.normal(0, sigma, 1)
        
    return (x,y)
    
    
def gendata_1d(fun, min, max, n=100, sigma=1, seed=None):
    '''
    '''
    if min > max:
        raise ValueError('min must be less than max')
    if sigma < 0:
        raise ValueError('sigma must be positive')
    
    if seed != None:
        np.random.seed(seed)
    
    x = np.random.uniform(min, max, n) 
    y = fun(x) + np.random.normal(0, sigma, n)
    return (x,y)
    
def f1(x):
    x = np.ravel(np.asarray(x))
    return np.sin(x)/x

def f2(x):
    x = np.ravel(np.asarray(x))
    return 3.0*np.sin(x**2) + 2.0*np.sin(1.5*x+1)

if __name__ == '__main__':
    kernel = SEKernel(np.log(0.05), np.log(1))   
    #kernel = LinearKernel()
    #plot_gp_prior(kernel, -1, 1, n=5) 
    plot_cov(kernel, -1, 1)
    
    x = np.linspace(-1, 1, 10)
    x = x[:,np.newaxis]
    print kernel(x)
    print np.linalg.det(kernel(x))
    print np.linalg.eig(kernel(x))
    
    plot1d_kernel(kernel, -1, 1, 0.01, 5)
    
    
    
    