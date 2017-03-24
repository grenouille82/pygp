'''
Created on Apr 2, 2013

@author: marcel
'''
import numpy as np
from upgeo.util.stats import covcorr, corrcov
if __name__ == '__main__':
    
    R = np.array([[1.00, 0.25, 0.90], [0.25, 1.00, 0.50], [0.90, 0.50, 1.00]])
    print R 
    var = np.array([1, 4, 9])
    S = corrcov(R, var)
    print 'S={0}'.format(S)
    R1 = covcorr(S)
    print R1