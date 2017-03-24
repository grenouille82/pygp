'''
Created on Mar 17, 2011

@author: marcel
'''
import numpy as np

from scikits.learn.base import BaseEstimator, RegressorMixin
from upgeo.util.exception import NotFittedError
from scipy.spatial.kdtree import KDTree

class KNNRegression(BaseEstimator, RegressorMixin):
    
    __slots__ = ('__k', '__data', '__target', '__use_kdtree', '_is_init')
    
    def __init__(self, k=1, use_kdtree=True):
        '''
        '''
        self.__k = k
        self.__use_kdtree = use_kdtree
        self._is_init = False
    
    def fit(self, X, y):
        '''
        '''
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise TypeError('X must be two dimensional')
        if y.ndim != 1:
            raise TypeError('y must be one dimensional')
        
        self.__data = KDTree(X) if self.__use_kdtree else X
        self.__target = y
        
        self._is_init = True
    
    def predict(self, X):
        '''
        '''
        self.__init_check()
        X = np.asarray(X)
        

    def _get_number_of_neighbors(self):
        '''
        '''
        return self.__k
    
    k = property(fget=_get_number_of_neighbors)
    
    def __init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')
