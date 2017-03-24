'''
Created on Mar 23, 2011

@author: marcel
'''
import numpy as np

from abc import ABCMeta, abstractmethod

from scikits.learn.base import BaseEstimator, RegressorMixin

from upgeo.regression.function import GaussianBasis, RadialBasisFunction
from upgeo.util.metric import tspe
from upgeo.util.exception import NotFittedError

class RBFNetwork(BaseEstimator, RegressorMixin):
    '''
    classdocs
    @todo:
    - allow the choice between normalized and unnormalized outputs
    '''
    __metaclass__ = ABCMeta
    
    __slots__ = ('_weights', '_kernel', '_rbf', '_is_init', '_bias')

    def __init__(self, kernel=GaussianBasis(1.0), bias=True):
        '''
        Constructor
        @todo:
        - check whether kernel is a subclass of RBFTypes, not an instance
        '''
        
        self._kernel = kernel
        self._is_init = False
        
    def fit(self, X, y):
        ''' 
        '''
        X = np.atleast_2d(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise TypeError('X must be two dimensional')
        if y.ndim != 1:
            raise TypeError('y must be one dimensional')
        
        #init and learn the hidden layer representation of rbf-net
        self._rbf = self._learn_rbf(X) 
        #learn the output weights of the rbf-net
        Z = self._map_rbf(X)
        self._weights = self._learn_weights(Z, y)
        
        self._init = True
        
        #predict network error
        yhat = self.predict(X)
        e = tspe(y, yhat)
        return e
    
    def predict(self, X):
        '''
        '''
        self._init_check()
        
        X = np.atleast_2d(X)
        if X.ndim != 2:
            raise TypeError('X must be a 2-dimensional array.')
        
        Z = self._map_rbf(X)
        y = np.dot(Z, self._weights)
        return y
    
    @abstractmethod
    def _learn_rbf(self, X, y):
        pass 
    
    @abstractmethod
    def _learn_weights(self, X, y):
        pass
    
    @abstractmethod
    def _map_rbf(self, X):
        pass
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')
        
class SimpleRBFNetwork(RBFNetwork):
    '''
    @todo: 
    - validate init_centers
    - parametrize init_centers
    '''
    __slots__ = ('_rho', '_init_centers')
    
    def __init__(self, kernel=GaussianBasis(1.0), bias=True, rho=1.0, init_centers='data'):
        RBFNetwork.__init__(self, kernel, bias)
        self._rho = rho
        self._init_centers = init_centers
      
    def _learn_rbf(self, X, y):
        '''
        '''
        #initialize candidate basis functions
        func_name = 'make_rbf_from_' + self._init_centers
        func = getattr(RadialBasisFunction, func_name)
        rbf = func(self._kernel, X)
        return rbf 
    
    def _learn_weights(self, X, y):
        
        pass
    
    def _map_rbf(self, X):
        '''
        '''
        Z = self._rbf(X)
        if self._bias:
            m = Z.shape[0]
            Z = np.c_[np.ones(m), Z]
            
        return Z

class SelectiveRBFNetwork(RBFNetwork):
    '''
    RBF Network learning algorithm represented by [Liao nad Carin] which is a 
    simplification of their multitask rbf-network. The algorithm inremental selects
    basis function until the error function doesn't increase dramatically.
    
    @todo: 
    - validate init_centers
    - parametrize init_centers
    '''
    __slots__ = ('_rho', '_init_centers')
    
    def __init__(self, kernel=GaussianBasis(1.0), bias=True, rho=1.0, init_centers='data'):
        RBFNetwork.__init__(self, kernel, bias)
        self._rho = rho
        self._init_centers = init_centers
      
    def _learn_rbf(self, X, y):
        '''
        '''
        #initialize candidate basis functions
        func_name = 'make_rbf_from_' + self._init_centers
        func = getattr(RadialBasisFunction, func_name)
        candidate_rbf = func(self._kernel, X)
        
        Z = None 
    
    def _learn_weights(self, X, y):
        pass
    
    def _map_rbf(self, X, rbf=None):
        '''
        '''
        if rbf == None:
            rbf = self._rbf
            
        Z = rbf(X)
        if self._bias:
            m = Z.shape[0]
            Z = np.c_[np.ones(m), Z]
            
        return Z
    