'''
Created on Mar 9, 2012

@author: marcel
'''

import numpy as np
import time

from upgeo.base.gp import SparseGPRegression
from upgeo.base.kernel import ARDSEKernel, NoiseKernel
from upgeo.base.infer import FITCOnePassInference
from upgeo.base.selector import RandomSubsetSelector

if __name__ == '__main__':
    
    X = np.random.randn(10000,13)
    y = np.random.randn(10000)
    
    kernel = ARDSEKernel(np.log(1)*np.ones(13), np.log(2)) + NoiseKernel(np.log(1))
    gp = SparseGPRegression(kernel, infer_method=FITCOnePassInference, selector=RandomSubsetSelector(30))
    gp.fit(X, y)
    likel_fun = gp.likel_fun
    
    t = time.time()
    print 'likel={0}'.format(likel_fun())
    print 'time={0}'.format(time.time()-t)
    t = time.time()
    print 'grad={0}'.format(likel_fun.gradient())
    print 'time={0}'.format(time.time()-t)
    
    #t = time.time()
    #for i in xrange(50):
    #    likel_fun.gradient()
    #print 'time={0}'.format(time.time()-t)
    