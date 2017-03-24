'''
Created on Feb 28, 2011

@author: marcel
'''

import numpy as np
 
from scikits.learn.base import BaseEstimator 

from upgeo.util.exception import NotFittedError
from upgeo.util.metric import distance_matrix
from upgeo.util.array import unique2d

class KMeans(BaseEstimator):
    '''
    classdocs
    
    @todo: 
    ------
    - define a iterator over clusters
    - extract/implement an own base/meta class (because BaseEstimator looks at 
      the arguments of to constructor to declare the class attributes)
    - restrict the number of cluster. should depend on the dataset size
    
    '''
    __slots__ = ('__ncluster',              #number of clusters in the clustering
                 '__nfeatures',             #dimension of the feature space
                 '__centers',               #estimated cluster centers
                 '__k',                     #max desired clusters
                 '__allow_empty_cluster',   #flag allowing the removal of empty clusters
                 '_is_init'                 #flag checking for model fitting
                 )                

    def __init__(self, k, empty_cluster=False):
        '''    
        Constructor
        '''
        assert k > 0, 'number of clusters must be greater than 0'
        
        self.__k = k
        self.__allow_empty_cluster = empty_cluster
        self._is_init = False
        
        
    def fit(self, x, k=None, init='random', max_iter=100, max_runs=1):
        '''
        TODO:
        -----
        - allow the parametrization of initialization methods by functions
        - check if error estimation works correctly
        - use error tolerance based termination criterion
        '''
        x = np.asarray(x)
        n,d = x.shape
        
        if k is None:
            k = self.__k
        else:
            self.__k = k
            
        if hasattr(init, '__array__'):
            #init is passed as a set of initial clusters, number of runs are 
            #ignored
            init = np.asarray(init)
            max_runs = 1
            
        del_empty = ~self.__allow_empty_cluster
        
        best_centers = None
        best_labels  = None
        best_error = np.infty
        for i in xrange(max_runs):
            if init == "kcenters":
                centers = _init_kcenters(x, k)
            elif init == "random":
                centers = _init_random(x, k)
            elif hasattr(init, '__array__'):
                centers = _init_from_array(x, k, init)
            else:
                raise TypeError('bad init parameter: {0}'.format(init))
    
            old_labels = np.ones(n)*-1
            for j in xrange(max_iter):
                print j
                labels, dist = _predict_clusters(x, centers)
                centers = _estimate_centers(centers, x, k, labels, del_empty)
                
                if not np.any(labels != old_labels):
                    break
                old_labels = labels 
            
            print labels.shape
            print dist.shape
            #print dist
            error = np.sum(dist[:,labels]**2)
            if error < best_error:
                best_centers = centers
                best_labels = labels
                best_error = error
            
        self._is_init = True
        self.__centers = best_centers
        self.__ncluster = len(best_centers)
        self.__nfeatures = d
        self.__fit_error = best_error
        
        
        return best_labels, best_error
    
    def predict(self, x):
        '''
        TODO:
        -----
        - check if error computation works correctly
        '''
        self.__init_check()
        
        x = np.asarray(x)
        assert x.shape[1] == self.__nfeatures, 'x must have shape(:,nfeatures)'
        
        label, dist = _predict_clusters(x, self.__centers)
        error = np.sum(dist[:,label]**2)
        return label, error
    
    def get_cluster_center(self, i):
        '''
        '''
        self.__init_check()
        self.__range_check(i)
        return self.__centers[i]
        
    def _get_k(self):
        '''
        '''
        return self.__k
    
    def _get_ncluster(self):
        '''
        '''
        self.__init_check()
        return self.__ncluster
    
    ncluster = property(fget=_get_ncluster)
    
    def _get_nfeatures(self):
        '''
        '''
        self.__init_check()
        return self.__nfeatures
    
    nfeatures = property(fget=_get_nfeatures)
    
    def _get_centers(self):
        '''
        '''
        self.__init_check()
        return self.__centers;
    
    centers = property(fget=_get_centers)
    
    def _get_fit_error(self):
        '''
        '''
        self.__init_check()
        return self.__fit_error
    
    fit_error = property(fget=_get_fit_error)
    
    def __range_check(self, index):
        '''
        Check whether the specified index is in range.
        '''
        if index < 0 or index > self.__k:
            raise IndexError("Index '{0}' out of bounds".format(index))
        
    def __init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked')
    

def _predict_clusters(x, centers):
    '''
    '''
    x = np.asarray(x)
    centers = np.asarray(centers)
    
    dist = distance_matrix(x, centers, metric='euclidean')
    labels = np.argmin(dist, 1)
    return labels, dist

def _estimate_centers(centers, x, k, labels, del_empty=False):
    '''
    '''    
    empty_cluster = np.zeros(len(centers), dtype='bool')
    
    for i in xrange(k):
        cluster = x[labels==i,:]
        if len(cluster) > 0:
            centers[i] = np.mean(x[labels==i,:], axis=0)
        elif not del_empty:
            empty_cluster[i] = True
    
    return centers[~empty_cluster]

def _init_kcenters(x, k):
    '''
    '''
    raise NotImplementedError('kcentres is not supported yet')

def _init_random(x, k):
    '''
    '''
    x = unique2d(x)
    n = x.shape[0]
    
    seeds = np.random.permutation(n)[:k]
    centers = x[seeds]
    return centers
            
def _init_from_array(x, k, centers):
    '''
    '''
    centers = np.asarray(centers)
    assert centers.shape == (k, x.shape[1])
    return centers.copy()
