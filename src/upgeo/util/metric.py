'''
Created on Mar 2, 2011

@author: marcel

TODO
----
- construct a class hierarchy for distance matrix computation
- allow caching and other fancy stuff
'''
import numpy as np


from abc import ABCMeta, abstractmethod

from numpy.core.numeric import inf
from scipy.spatial.kdtree import KDTree
from scipy.spatial.distance import cdist


def distance_matrix(x, y=None, metric='euclidean', p=2, V=None, VI=None, w=None):
    '''
    Computes the pairwise distance between m original observation or the distance 
    of pairs in a cartesian product in a n-dimensional space. A full distance 
    matrix is returned. This function wraps the functionality of both functions in 
    package. For details of the parametrizing see.
    
    Parameter
    ---------
    
    Return
    ------
    
    @todo
    - 
    
    '''
    x = np.atleast_2d(x)
    y = x if y is None else np.atleast_2d(y)
    
    distm = cdist(x, y, metric=metric, p=p, V=V, VI=VI, w=w)
    #distm = squareform(distm, checks=False)
    return np.squeeze(distm)

def tse(y, mu):
    '''
    Calculate the total squared error of mean estimator mu given the observation x.
    '''
    y = np.asarray(y)
    if y.ndim != 1:
        raise TypeError('dimension of x must be 1')

    e = np.sum((y-mu)**2.0) 
    return e

def mse(y, mu):
    '''
    Calculate the mean squared error of mean estimator mu given the observation x.
    '''
    if y.ndim != 1:
        raise TypeError('dimension of x must be 1')

    y = np.asarray(y)
    n = y.size
    e = tse(y, mu) / n
    return e

def tspe(y, yhat):
    '''
    Calculate the total squared prediction error between the true value x and
    the predicted value y.
    '''
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    
    if y.ndim != 1 or yhat.ndim != 1:
        raise TypeError('dimension of x and y must be 1')
    
    e = np.sum((y-yhat)**2.0)
    return e




def mspe(y, yhat):
    '''
    Calculate the mean squared prediction error between the true value x and 
    the predicted value y.
    '''
    n = y.size
    e = tspe(y, yhat) / n
    return e

def nlpp(y, mu, var):
    n = y.size
    return np.sum(((mu-y)**2)/(2.0*var) + 0.5*np.log(2.0*np.pi*var))/n    

def nlp(y, yhat, var):
    n = y.size
    return np.sum(((yhat-y)**2)/(2.0*var) + 0.5*np.log(2.0*np.pi*var))/n



def rsquare(y, yhat):
    '''
    @todo: optimize code
    '''
    rss = tspe(y, yhat)
    tss = tse(y, np.mean(y))
    
    print 'rss={0}, tss={1}, ess={2}'.format(rss, tss, tse(yhat, np.mean(y)))
    
    r = 1.0 - rss/tss
    return r

class NearestNeighborSearch:
    '''
    @todo: 
    - mask attributes(features) of the data vectors
    - the methods query_ball and query_points in subclasses should return 
      numpy array types
    - exception handling should be identical in all sub classes
    - return values should be the same in all sub classes
    '''
    __metaclass__ = ABCMeta
    
    __slots__ = ('_data', '_ndim')
    
    def __init__(self, data):
        '''
        @todo:
        - check dimension
        '''
        self._data = np.asarray(data).copy()
        self._ndim = self._data.ndim
        
    @abstractmethod
    def query_knn(self, x, k=1, p=2.0, distance_upper_bound=inf):
        pass
    
    @abstractmethod
    def query_ball(self, x, r, p=2.0):
        pass 
    
    @abstractmethod
    def query_pairs(self, r, p=2.0):
        pass
    
    def data_point(self, i):
        '''
        '''
        return self.__data[i]
    
    def _get_data(self):
        '''
        '''
        return self.__data.copy()
    
    data = property(fget=_get_data)
    
    def _get_ndim(self):
        '''
        '''
        return self.__ndim

    ndim = property(fget=_get_ndim)

class KDTreeSearch(NearestNeighborSearch):
    '''
    '''
    __slots__ = ('__tree')
    
    def __init__(self, data):
        NearestNeighborSearch.__init__(self, data)
        self.__tree = KDTree(self._data)
    
    def query_knn(self, x, k=1, p=2.0, distance_upper_bound=inf):
        '''
        '''
        distances, neighbors = self.__tree.query(x, k, p, distance_upper_bound)
        return neighbors, distances
    
    def query_ball(self, x, r, p=2.0):
        '''
        '''
        return self.__tree.query_ball(x, r, p)
    
    def query_pairs(self, r, p=2.0):
        '''
        '''
        return self.__tree.query_pairs(r, p)


class NaiveMatrixSearch(NearestNeighborSearch):
    '''
    '''
    __slots__ = ()
    
    def query_knn(self, x, k=1, p=2.0, distance_upper_bound=inf):
        '''
        '''
        distm = distance_matrix(x, self._data, p=p)
        perm = np.argsort(distm, 0)
        
        neighbors = perm[:k]
        distances = distm[perm[:k]]
        if distance_upper_bound < inf:
            bound_mask = distances > distance_upper_bound
            distances[bound_mask] = inf
            
        return neighbors, distances
    
    def query_ball(self, x, r, p=2.0):
        '''
        @todo:
        - should the method return a distance-sorted list of the neighborhood? 
        '''
        x = np.asarray(x)
        [n,m] = x.shape()
        
        if r < 0:
            raise ValueError('radius r must be non-negative: {0}'.format(r))
        if m != self.ndim:
            raise ValueError('dimension of x and the search structure must be equal')
           
        distm = distance_matrix(x, self._data, p=p)
        region_mask = distm <= r
        
        neighbors = np.empty(n, dtype='object')
        for i in xrange(n):
            region,_ = np.where(region_mask[i])
            neighbors[i] = region.tolist()
               
        return neighbors
    
    def query_pairs(self, r, p=2.0):
        '''
        '''
        if r < 0:
            raise ValueError('radius r must be non-negative: {0}'.format(r))
        
        distm = distance_matrix(self._data, p=p)
        i,j = np.where(distm <= r)
        pairs = set(zip(i.tolist(), j.tolist()))
        return pairs

if __name__ == '__main__':
    X = np.random.rand(100,3)
    tree = KDTreeSearch(X)