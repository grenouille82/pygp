'''
Created on Mar 14, 2011

@author: marcel
'''

import numpy as np

from abc import ABCMeta, abstractmethod
from scikits.learn.base import BaseEstimator, RegressorMixin

from upgeo.util.exception import NotFittedError
from scikits.learn.cross_val import KFold
from upgeo.regression.function import PolynomialFunction, RadialBasisFunction
from upgeo.util.metric import mspe
from upgeo.util.cluster import KMeans

class LinearRegressionModel(BaseEstimator, RegressorMixin):
    '''
    @todo:
    - allow fitting of multiple outputs
    - implement a summary statistic of the fitted model
    '''
    __slots__ = ('_intercept', 
                 '_weights', 
                 '_is_init', 
                 '_basis_function',
                 '_X',
                 '_y')
    
    __metaclass__ = ABCMeta
    
    def __init__(self, basis_function=None):
        self._basis_function = basis_function
        self._is_init = False
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    def predict(self, X, ret_var=False):
        '''
        '''
        self._init_check()
        X = self._preprocess_data(X)
        yhat = np.dot(X, self._weights) + self._intercept
        #yhat = np.dot(X, self._weights)
        if ret_var:
            #estimate the variance using residuals of the training set
            Xtrain = self._X
            ytrain = self._y
            n = len(ytrain)
            ythat = np.dot(Xtrain, self._weights) + self._intercept
            resid = ytrain-ythat
            var = np.sum(resid**2.0) / (n-2.0)
            var = np.ones(len(X))*var
            print 'var={0}'.format(var)
            return yhat, var
             
        return yhat
    
    def _get_intercept(self):
        '''
        '''
        self._init_check()
        return self._intercept
    
    intercept = property(fget=_get_intercept)
    
    def _get_weights(self):
        '''
        '''
        self._init_check()
        return self._weights.copy()
    
    weights = property(fget=_get_weights)
    
    def _get_basis_function(self):
        '''
        '''
        return self._basis_function
    
    basis_function = property(fget=_get_basis_function)
    
    def _preprocess_data(self, X):
        X = np.asarray(X)
        return X if self._basis_function == None else self._basis_function(X)
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')

class LSRegresion(LinearRegressionModel):
    '''
    '''
    __slots__ = ()
    
    def fit(self, X, y, preprocess=True):
        X = self._preprocess_data(X)
        y = np.asarray(y)
        self._X = X
        self._y = y
        
        if X.ndim != 2:
            raise TypeError('X must be two dimensional')
        if y.ndim != 1:
            raise TypeError('y must be one dimensional')
        
        [m,d] = X.shape
        if m <= d:
            raise ValueError('')
        
        X = np.c_[np.ones(m), X]
        A = np.linalg.pinv(X)
        beta = np.dot(A,y)
        
        self._intercept = beta[0]
        self._weights   = beta[1:]
        self._is_init = True
            
class RidgeRegression(LinearRegressionModel):
    '''
    '''
    __slots__ = ('_alpha') 
    
    def __init__(self, alpha=1.0, basis_function=None):
        '''
        ''' 
        LinearRegressionModel.__init__(self, basis_function)
        if alpha < 0.0:
            raise ValueError('alpha must be non-negative')
        
        self._alpha = alpha
        
    def fit(self, X, y):
        '''
        @todo:
        - allow multiple output fitting
        '''
        X = self._preprocess_data(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise TypeError('X must be two dimensional')
        if y.ndim != 1:
            raise TypeError('y must be one dimensional')
        
        [m,d] = X.shape
        
        X = np.c_[np.ones(m), X] 
        if self._alpha != 0:
            #if penalty term is used, the intercept and the ridge coefficients 
            #is estimated by using the technique of data augmentation, 
            #therefore the problem is recasted in least square problem
            lam = np.diag(np.r_[0, np.ones(d)]*np.sqrt(self._alpha))
            X = np.r_[X, lam]
            y = np.r_[y, np.zeros(d+1)]
            #lam = np.diag(np.r_[np.ones(d)]*np.sqrt(self._alpha))
            #X = np.r_[X, lam]
            #y = np.r_[y, np.zeros(d)]
                   
        #A = np.linalg.pinv(X)
        #beta = np.dot(A,y)
        beta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
         
        self._intercept = beta[0]
        self._weights   = beta[1:]
        #self._weights   = beta[0:]
        self._is_init = True
        
    def _get_alpha(self):
        return self.__alpha
    
    alpha = property(fget=_get_alpha)
    
class OptPolynomialRegression(LinearRegressionModel):
    '''
    @todo: - basis function cannot set after initialization of the regression model.
             to overcome this situation, make  the setter public or use a factory
             pattern or the parametrized regression model haven't been initialized
             with a basis function (is the case yet).
           - mask features for the polynom function
           - include variance in the model selection
    '''
    __slots__ = ('_n_min',      #min polynom deg of the search space 
                 '_n_max',      #max polynom deg of the search space
                 '_n_opt',      #estimated opt deg via cross validation
                 '_reg_model',  #baseline regression model
                 '_n_folds'     #number of folds for the cross validation
                 )
    
    def __init__(self, reg_model, n_min, n_max, nfolds=10):
        LinearRegressionModel.__init__(self)
        if n_min > n_max:
            raise ValueError('n_min must be smaller or equal than n_max')
        if nfolds < 1:
            raise ValueError('nfolds must be greater than zero.')
        
        if reg_model.basis_function != None:
            raise ValueError('basis function of regression must be uninitialized')
        
        self._n_min = n_min
        self._n_max = n_max
        self._nfolds = nfolds
        self._reg_model = reg_model
        
    def fit(self, X, y):
        '''
        '''
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise TypeError('X must be two dimensional')
        if y.ndim != 1:
            raise TypeError('y must be one dimensional')
        
        m = X.shape[0]
        
        #optimize the degree of polynom
        loo = KFold(m, self._nfolds) #cross validation is the same for each degree of polynom
        eval_stats = np.ones((self._n_max-self._n_min+1, 2))*np.nan #holds the mean and the variance of mspe
        i = 0
        for n in xrange(self._n_min, self._n_max+1):
            polynom = PolynomialFunction(n)
            Xp = polynom(X)
            
            errors = np.zeros(self._nfolds)
            j = 0
            for train, test in loo:
                self._reg_model.fit(Xp[train,:], y[train])
                yhat = self._reg_model.predict(Xp[test,:])
                errors[j] = mspe(y[test], yhat) 
                j += 1
                
            eval_stats[i,0] = np.mean(errors)
            eval_stats[i,1] = np.var(errors)
            i += 1
        
        self._n_opt = np.argmin(eval_stats[:,0]) + self._n_min
        self._basis_function = PolynomialFunction(self._n_opt)
        
        #learn the regression parameters
        X = self._preprocess_data(X)
        self._reg_model.fit(X, y)
        self._intercept = self._reg_model.intercept
        self._weights = self._reg_model.weights
        
        self._is_init = True
        
    def _get_nfolds(self):
        return self._nfolds
    
    nfolds = property(fget=_get_nfolds)
    
    def _get_n_opt(self):
        self._init_check()
        return self._n_opt
    
    n_opt = property(fget=_get_n_opt)
    
    def _get_reg_model(self):
        return self._reg_model
    
    reg_model = property(fget=_get_reg_model)
        

class OptRBFRegression(LinearRegressionModel):
    '''
    @todo: - basis function cannot set after initialization of the regression model.
             to overcome this situation, make  the setter public or use a factory
             pattern or the parametrized regression model haven't been initialized
             with a basis function (is the case yet).
    '''
    
    slots = ('_kernel', '_param_grid', '_data_ratio', '_nfolds')
    
    def __init__(self, reg_model, kernel, param_grid, data_ratio=None, nfolds=10):
        '''
        @todo: - check if kernel is a class not an instance and it is paramtrizeable
        '''
        LinearRegressionModel.__init__(self)
        if data_ratio < 0 or data_ratio > 0.2:
            raise ValueError('data_ratio must be in interval [0,0.2].')
        if nfolds < 1:
            raise ValueError('nfolds must be greater than zero.')
        
        if reg_model.basis_function != None:
            raise ValueError('basis function of regression must be uninitialized')
        
        self._reg_model = reg_model
        
        self._kernel = kernel
        self._param_grid = np.asarray(param_grid)
        self._data_ratio = data_ratio
        self._nfolds = nfolds
        
    def fit(self, X, y):
        '''
        '''
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise TypeError('X must be two dimensional')
        if y.ndim != 1:
            raise TypeError('y must be one dimensional')
        
        m = X.shape[0]
        
        #optimize the kernel parameters
        #todo: assumption is here just the kernel have one parameter to optimize
        loo = KFold(m, self._nfolds) #cross validation is the same for each degree of polynom
        n = self._param_grid.size
        eval_stats = np.ones((n,2))*np.nan#holds the mean and the variance of mspe
        
        #determine the rbf-centers for each fold
        fold_centers = []
        for train,_ in loo:
            fold_centers.append(self._determine_data_centers(X[train]))
            
        i = 0
        for param in self._param_grid: 
            errors = np.zeros(self._nfolds)
            j = 0
            for train, test in loo:
                bf = self._make_basis_function(fold_centers[j], param)
                Xi = bf(X)
            
                self._reg_model.fit(Xi[train,:], y[train])
                yhat = self._reg_model.predict(Xi[test,:])
                errors[j] = mspe(y[test], yhat) 
                j += 1
                
            eval_stats[i,0] = np.mean(errors)
            eval_stats[i,1] = np.var(errors)
            i += 1
        
        self._param_opt = self._param_grid[np.argmin(eval_stats[:,0])]
        kernel = self._kernel(self._param_opt)
        if self._data_ratio == None:
            self._basis_function = RadialBasisFunction.make_rbf_from_data(X, kernel)
        else:
            self._basis_function = RadialBasisFunction.make_rbf_from_kmeans(X, kernel, 
                                                                            self._data_ratio)
        
        #learn the regression parameters
        X = self._preprocess_data(X)
        self._reg_model.fit(X, y)
        self._intercept = self._reg_model.intercept
        self._weights = self._reg_model.weights
        
        self._is_init = True

    def _make_basis_function(self, centers, param):
        kernel = self._kernel(param)
        return RadialBasisFunction(kernel, centers)
    
    def _determine_data_centers(self, X):
        centers = X
        if self._data_ratio !=  None:
            k = np.int(np.floor(len(X)*self._data_ratio))
            kmeans = KMeans(k)
            kmeans.fit(X)
            centers = kmeans.centers
        return centers
    
    def _get_nfolds(self):
        return self._nfolds
    
    nfolds = property(fget=_get_nfolds)
    
    def _get_param_opt(self):
        self._init_check()
        return self._param_opt
    
    param_opt = property(fget=_get_param_opt)
    
    def _get_reg_model(self):
        return self._reg_model
    
    reg_model = property(fget=_get_reg_model)


class KernelRegression(LinearRegressionModel):
    pass
        
class LassoRegression(LinearRegressionModel):
    pass    