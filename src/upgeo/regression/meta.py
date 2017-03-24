'''
Created on Mar 1, 2012

@author: marcel
'''
from upgeo.util.exception import NotFittedError
from scikits.learn.base import RegressorMixin, BaseEstimator
from sklearn.decomposition.pca import ProbabilisticPCA
from sklearn.decomposition.kernel_pca import KernelPCA

class PCARegression(BaseEstimator, RegressorMixin):
    '''
    '''
    __slots__ = ('_dim',
                 '_reg_model',
                 '_pca_model', 
                 '_is_init' )
    
    def __init__(self, reg_model, dim=None):
        self._dim = dim
        self._reg_model = reg_model
        self._is_init = False
    
    def fit(self, X, y):
        
        dim = self._dim
        reg_model = self._reg_model
        
        if dim == None:
            pca_model = ProbabilisticPCA('mle')
        else:
            pca_model = ProbabilisticPCA(dim)
        
        Z = pca_model.fit_transform(X)
        print 'shape1={0}'.format(Z.shape)
        reg_model.fit(Z,y)
        
        self._pca_model = pca_model
        self._is_init = True
    
    def predict(self, X):
        '''
        '''
        self._init_check()
        Z = self._pca_model.transform(X)
        yhat = self._reg_model.predict(Z) 
        return yhat
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')

class KernelPCARegression(BaseEstimator, RegressorMixin):
    '''
    '''
    __slots__ = ('_dim',
                 '_kernel',
                 '_degree',
                 '_reg_model',
                 '_pca_model', 
                 '_is_init' )
    
    def __init__(self, reg_model, dim, kernel='rbf', degree=3):
        self._dim = dim
        self._kernel = kernel
        self._degree = degree
        self._reg_model = reg_model
        self._is_init = False
    
    def fit(self, X, y):
        
        dim = self._dim
        kernel = self._kernel
        degree = self._degree
        reg_model = self._reg_model
        
        pca_model = KernelPCA(n_components=dim, kernel=kernel, degree=degree)
        print dim
        Z = pca_model.fit_transform(X)
        print 'shape={0}'.format(Z.shape)
        print pca_model.transform(X).shape
        reg_model.fit(Z,y)
        
        self._pca_model = pca_model
        self._is_init = True
    
    def predict(self, X):
        '''
        '''
        self._init_check()
        Z = self._pca_model.transform(X)
        yhat = self._reg_model.predict(Z) 
        return yhat
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')

