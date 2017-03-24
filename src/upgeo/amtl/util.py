'''
Created on May 21, 2012

@author: marcel
'''

import numpy as np

def gendata_fmtlgp_1d(pkernel, skernel, min, max, n=100, ntasks=1, sigma=0, seed=None):
    '''
        n: number of samples for each task
        ntasks: number of secondary tasks
    '''
    if min > max:
        raise ValueError('min must be less than max')
    if sigma < 0:
        raise ValueError('sigma must be >= 0.')

    if seed != None:
        np.random.seed(seed)


    x = np.linspace(min, max, n)
    x = x[:,np.newaxis]

    Kp = pkernel(x)
    Ks = skernel(x)
    yp = np.random.multivariate_normal(np.zeros(n), Kp)
    if sigma > 0:
        yp += np.random.normal(0, sigma, n)
        
    ys = np.zeros((n, ntasks))
    alpha = np.random.randn(ntasks)
    beta = np.random.rand(ntasks)
    for i in xrange(ntasks):
        alpha[i]
        beta[i]
        ys[:,i] = np.random.multivariate_normal(alpha[i]*yp, beta[i]*Ks)
        #ys[:,i] = np.random.multivariate_normal(alpha[i]*yp, beta[i]*Ks)
        if sigma > 0:
            ys[:,i] += np.random.normal(0, sigma, n)
    #ys = np.tile(yp[:,np.newaxis], (1,ntasks))
    return x, yp, ys

def gendata_fmtlgp_2d(pkernel, skernel, min, max, n=100, ntasks=1, sigma=0, seed=None):
    '''
        n: number of samples for each task
        ntasks: number of secondary tasks
    '''
    if min > max:
        raise ValueError('min must be less than max')
    if sigma < 0:
        raise ValueError('sigma must be >= 0.')

    if seed != None:
        np.random.seed(seed)


    x1 = np.linspace(min, max, n)
    x2 = np.linspace(min, max, n)
    G,H = np.meshgrid(x1, x2)
    x = np.c_[np.ravel(G), np.ravel(H)]
    print 'shape'
    print x.shape

    Kp = pkernel(x)
    Ks = skernel(x)
    yp = np.random.multivariate_normal(np.zeros(n*n), Kp)
    if sigma > 0:
        yp += np.random.normal(0, sigma, n*n)
        
    ys = np.zeros((n*n, ntasks))
    alpha = np.random.randn(ntasks)
    beta = np.random.rand(ntasks)
    for i in xrange(ntasks):
        alpha[i]
        beta[i]
        ys[:,i] = np.random.multivariate_normal(alpha[i]*yp, beta[i]*Ks)
        #ys[:,i] = np.random.multivariate_normal(alpha[i]*yp, beta[i]*Ks)
        if sigma > 0:
            print 'jdsdihfishfd'
            ys[:,i] += np.random.normal(0, sigma, n*n)
    ys = np.tile(yp[:,np.newaxis], (1,ntasks))
    return x, yp, ys


import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    from upgeo.base.util import gendata_gp_1d
    from upgeo.base.kernel import SEKernel, NoiseKernel

    
    kernel = SEKernel(np.log(1), np.log(5))# + NoiseKernel(np.log(1))
    x,y = gendata_gp_1d(kernel, -5, 5, 100, 0.0)
    
    plt.plot(x,y)
    plt.show()
    
    ntasks = 5
    x, yp, ys = gendata_fmtlgp_1d(kernel, kernel, -5, 5, 100, ntasks, 0.0)
    plt.plot(x,yp)
    plt.show()
    for i in xrange(ntasks):
        plt.plot(x,ys[:,i])
        plt.show()
    
    step = 0.01
    x = np.arange(-10,10, step)
    y = np.zeros(len(x))
    y[np.all(np.c_[x > -0.5, x < 0.5],1)] = 1
    z = np.ones(np.sum(y == 1))
    y = np.sin(x)*1.2 + 0.4*x**2 + np.random.randn(len(x))*0.1
    
    print np.all(np.c_[x > -0.5, x < 0.5],1)
    print x
    print y
    print z
    z = np.hamming(9)
    yz = np.convolve(z,y, 'same')
    
    plt.plot(x, y)
    plt.plot(x, yz)
    plt.show()
    
    t=np.arange(0.1,20,0.01)
    x=0.5*t+ + 1.2*t+np.log(t)*3.1+np.random.randn(len(t))*0.001
    y=smooth(x, 15, 'bartlett')

    print np.hanning(10)

    plt.plot(x)
    plt.plot(y)
    plt.show()
