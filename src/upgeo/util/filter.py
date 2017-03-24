'''
Created on Feb 14, 2012

@author: marcel
'''

import numpy as np

from abc import ABCMeta, abstractmethod

class DatasetFilter(object):
    '''
    classdocs
    todo: - test whether passed argument by filter procedure is changed, if we are using a mask,
    '''
    
    __metaclass__ = ABCMeta

    __slots__ = ('_mask')

    def __init__(self, mask=None):
        '''
        Constructor
        '''
        self._mask = mask
    
    @abstractmethod 
    def process(self, X, reuse_stats=False):
        pass
    
    @abstractmethod
    def invprocess(self, X):
        pass

class CompositeFilter(DatasetFilter):
    
    __slots__ = ('_filters')
    
    def __init__(self, filters):
        DatasetFilter.__init__(self)
        self._filters = filters

    def process(self, X, reuse_stats=False):
        filters = self._filters
        Y = X
        for f in filters: #problematic if we use numpy array
            Y = f.process(Y, reuse_stats)
        return Y

    def invprocess(self, X):
        filters = self._filters
        Y = X
        for f in reversed(filters): #problematic if we use numpy array
            Y = f.invprocess(Y)
        return Y

    
class StandardizeFilter(DatasetFilter):
    '''
    TODO: -check for precomputed data by invoking the inverse part or reusing stats
    '''
    __slots__ = ('_ddof',
                 '_mean',
                 '_std')
    
    def __init__(self, ddof=1, mask=None):
        DatasetFilter.__init__(self, mask)
        self._ddof = ddof
        
    
    def process(self, X, reuse_stats=False):
        mask = self._mask
        ddof = self._ddof
        X = np.atleast_2d(X)
        
                
        if reuse_stats == True:
            mean = self._mean
            std = self._std
        else:
            if mask == None:
                mean = np.mean(X, axis=0)
                std = np.std(X, axis=0, ddof=ddof)
            else:
                mean = np.mean(X[:,mask], axis=0)
                std = np.std(X[:,mask], axis=0, ddof=ddof)
            self._mean = mean
            self._std = std
        
        if mask == None:
            Y = (X-mean)/std
        else:
            Y = X.copy()
            Y[:,mask] = (X[:,mask]-mean)/std
            
        return Y
    
    def invprocess(self, X):
        mask = self._mask
        X = np.atleast_2d(X)
        
        mean = self._mean
        std = self._std
        
        if mask == None:
            Y = X*std+mean
        else:
            Y = X.copy()
            Y[:,mask] = X[:,mask]*std+mean
        return Y
    
    def _get_mean(self):
        return self._mean
    
    mean = property(fget=_get_mean)
    
    def _get_std(self):
        return self._std
    
    std = property(fget=_get_std)
        
class MeanShiftFilter(DatasetFilter):
    '''
    TODO: -check for precomputed data by invoking the inverse part or reusing stats
    '''
    
    __slots__ = ('_mean'
                 )
    
    def __init__(self, mask=None):
        DatasetFilter.__init__(self, mask)
        
    def process(self, X, reuse_stats=False):
        mask = self._mask
        #X = np.atleast_2d(X)
        if X.ndim == 1:
            X = X[:,np.newaxis]
        
        

        if reuse_stats == True:
            mean = self._mean
        else:
            mean = np.mean(X, axis=0)
            print 'mean'
            print mean
            self._mean = mean
        
        if mask == None:
            Y = X-mean 
        else:
            Y = X.copy()
            Y[:, mask] = X[:,mask]-mean 
        return Y
    
    def _get_mean(self):
        return self._mean

    mean = property(fget=_get_mean)

    def invprocess(self, X):
        mask = self._mask
        X = np.atleast_2d(X)
        
        mean = self._mean
        
        if mask == None:
            Y = X+mean
        else:
            Y = X.copy()
            Y[:,mask] = X[:,mask]+mean
        return Y
    
    
class MinMaxFilter(DatasetFilter):
    
    __slots__ = ('_min',
                 '_max',
                 '_mask')
    
    def __init__(self, mask=None):
        DatasetFilter.__init__(self)
        self._mask = mask
    
    def process(self, X, reuse_stats=False):
        mask = self._mask
        X = np.atleast_2d(X)
        
        if reuse_stats == True:
            min = self._min
            max = self._max
        else:
            if mask == None:
                min = np.min(X, axis=0)
                max = np.max(X, axis=0)
            else:
                min = np.min(X[:,mask], axis=0)
                max = np.max(X[:, mask], axis=0)
            self._min = min
            self._max = max
        
        d = max-min
        if mask == None:
            Y = (X-min)/d
        else:
            Y = X.copy()
            Y[:,mask] = (X[:,mask]-min)/d
        return Y    

    def invprocess(self, X):
        mask = self._mask
        X = np.atleast_2d(X)
        
        min = self._min
        max = self._max
        d = max-min
        
        if mask == None:
            Y = X*d+min
        else:
            Y = X.copy()
            Y[:,mask] = X[:,mask]*d+min
        return Y


    def _get_mean(self):
        return self._mean

    mean = property(fget=_get_mean)
    
    def _get_min(self):
        return self._min

    min = property(fget=_get_min)

    def _get_max(self):
        return self._max
    
    max = property(fget=_get_max)
        
    
class BinarizeFilter(DatasetFilter):
    
    '''
        @todo: - remove
    '''
    __slots__ = ()
    
    def init(self):
        DatasetFilter.__init__(self)
        
    def process(self, X):
        n, m = X.shape
        
        Y = np.empty((n,0), dtype=np.bool)
        for i in xrange(m):
            values = np.unique(X[:,i])
            k = len(values)
            Yb = np.zeros((n,k), dtype=np.bool)
            for j in xrange(k):
                Yb[:,j] = X[:,i] == values[j]
                
            Y = np.c_[Y,Yb]
                
        return Y
    
class FunctionFilter(DatasetFilter):
    
    __slots__ = ('_fun',
                 '_invfun',
                 '_mask')
    
    def __init__(self, fun, invfun, mask=None):
        DatasetFilter.__init__(self, mask)
        
        self._fun = fun
        self._invfun = invfun
        self._mask = mask
        
    def process(self, X, reuse_stats=False):
        mask = self._mask
        X = np.atleast_2d(X)
        if mask == None:
            Y = self._fun(X)
        else:
            Y = X.copy()
            Y[:,mask] = self._fun(X[:,mask])
            
        return Y
    
    def invprocess(self, X):
        mask = self._mask
        X = np.atleast_2d(X)
        if mask == None:
            Y = self._invfun(X)
        else:
            Y = X.copy()
            Y[:,mask] = self._invfun(X[:,mask])
            
        return Y
    

class ValueFilter(DatasetFilter):
    
    __slots__ = ('_i',
                 '_val')
    
    def __init__(self, index, value):
        DatasetFilter.__init__(self)
        
        self._i = index
        self._val = value
    
    def process(self, X):
        Y = X[X[:,self._i] == self._val]
        return Y
    
class IntervalFilter(DatasetFilter):
    __slots__ = ('_i',
                 '_min',
                 '_max')
    
    def __init__(self, index, min=-np.inf, max=np.inf):
        DatasetFilter.__init__(self)
        
        self._i = index
        self._min = min
        self._max = max
    
    def process(self, X):
        Y = X[np.logical_and(X[:,self._i] >= self._min, X[:,self._i] <= self._max)]
        return Y

    
    
class NaNInfFilter(DatasetFilter):
    __slots__ = ()
    
    
    def init(self):
        DatasetFilter.__init__(self)
        
    
    def process(self, X):
        pass
    

class BagFilter(DatasetFilter):
    __slots__ = ('_bag',
                 '_filter')
    
    def __init__(self, bag, filter):
        self._bag = bag
        self._filter = filter
    
    def process(self, X, reuse_stats=False):
        X = np.atleast_2d(X)
        filter = self._filter
        
        if reuse_stats:
            Y = filter.process(X, True)
        else:
            bag = self._bag
            _, uidx = np.unique(bag, return_index=True)
            filter.process(X[uidx]) #learn filter states
            Y = filter.process(X, True)
        return Y
    
    def invprocess(self, X):
        X = np.atleast_2d(X)
        filter = self._filter
        
        Y = filter.invprocess(X)
        return Y
            
                
    
        
        