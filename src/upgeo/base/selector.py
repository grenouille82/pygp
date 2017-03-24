'''
Created on Sep 1, 2011

@author: marcel
'''

import numpy as np

from abc import ABCMeta, abstractmethod
from upgeo.util.cluster import KMeans

class SelectionMethod(object):
    
    __metaclass__ = ABCMeta
    
    __slots__ = ()
    
    def __init__(self):
        pass
    
    @abstractmethod
    def apply(self, X, y):
        pass
    
class FixedSelector(SelectionMethod):
    
    __slots__ = ('_Xu'   #fixed set of inducing points
                 )
    
    def __init__(self, Xu):
        SelectionMethod.__init__(self)
        self._Xu = Xu
    
    def apply(self, X, y):
        return self._Xu
    
class KMeansSelector(SelectionMethod):
    
    __slots__ = ('_k',   #number of centers
                 '_include_target'
                 )
    
    def __init__(self, k, include_target=False):
        SelectionMethod.__init__(self)
        self._k = k
        self._include_target = include_target
    
    def apply(self, X, y):
        k = self._k
        d = X.shape[1]
        if self._include_target:
            X = np.c_[X,y]
        if len(X) > k: 
            kmeans = KMeans(k)
            kmeans.fit(X, max_runs=5)
            return kmeans.centers[:,0:d] if self._include_target else kmeans.centers
        else:
            return X[:,0:d] if self._include_target else X


class RandomSubsetSelector(SelectionMethod):
    
    __slots__ = ('_m'   #size of the subset to be selected
                 )
    
    def __init__(self, m):
        SelectionMethod.__init__(self)
        if m < 1:
            raise ValueError('m must be at least 1')
        self._m = m
    
    def apply(self, X, y):
        m = self._m
        n = len(X)
        
        if n < m:
            m = n
            #raise ValueError('dataset size must be at least {0}.'.format(m))
        
        perm = np.random.permutation(n)
        Xu = X[perm[0:m]]
        return Xu