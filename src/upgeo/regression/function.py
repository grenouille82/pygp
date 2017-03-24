'''
Created on Mar 16, 2011

@author: marcel
'''

import numpy as np

from abc import ABCMeta, abstractmethod

from upgeo.util.cluster import KMeans
from upgeo.util.metric import distance_matrix

class BasisFunction:
    '''
    '''
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __call__(self, x):
        pass


class PolynomialFunction(BasisFunction):
    '''
    @todo: - mask features which are not used in the polynom creation process
    '''
    __slots__ = ('__n')
    
    def __init__(self, n):
        self.__n = n

    def __call__(self, x):
        x = np.asarray(x)
        
        if x.ndim != 2:
            raise ValueError('2-dimensional x is expected')
        
        n = self.__n
        m,d = x.shape
        z = np.zeros((m, n*d))
        z[:,0:d] = x
        
        polynoms = np.arange(2, n+1)
        for i in polynoms:
            z[:,(i-1)*d:i*d] = x**i
        
        return z
    
    def _get_degree(self):
        return self.__n
    
    degree = property(fget=_get_degree)

class RadialBasisFunction(BasisFunction):
    '''
    @todo:
    - currently, the basis function are identical for each dimension, but the
      opposite effect could be desired, that means each dimension defines his 
      own basis function. For example the basis functions (gaussian case) are
      represented by cluster centers after kmeans was applied. Hereby, the
      variances term (sigma) can be represented by the cluster variance.
    '''
    
    __slots__ = ('__kernel', '__centers')

    @staticmethod
    def make_rbf_from_kmeans(x, kernel, ratio=0.05):
        '''
        '''
        if ratio < 0.0 or ratio > 0.2:
            raise ValueError('ratio must be in range [0,0.2].')
        
        x = np.atleast_2d(x)
        m = x.shape[0]
        k = np.int(np.floor(m*ratio))
        k = 1 if k < 1 else k
            
        kmeans = KMeans(k)
        kmeans.fit(x, k)
        centers = kmeans.centers
        
        return RadialBasisFunction(kernel, centers)

    @staticmethod
    def make_rbf_from_data(x, kernel):
        '''
        '''
        centers = np.atleast_2d(x)
        return RadialBasisFunction(kernel, centers) 

    def __init__(self, kernel, centers):
        '''
        @todo:
        - check subtype of the kernel parameter
        '''
        centers = np.atleast_2d(centers)
        if centers.ndim != 2:
            raise ValueError('2-dimensional centers is expected')
        
        self.__centers = centers
        self.__kernel = kernel
    
    def __call__(self, x):
        x = np.asarray(x)        
        distm = distance_matrix(x, self.__centers)
        z = self.__kernel(distm)
        return z
        
    def _get_number_of_centers(self):
        return self.__centers.shape[0]
    
    ndim = property(fget=_get_number_of_centers)
    
class RBFTypes:
    '''
    '''
    __metaclass__ = ABCMeta
    
    __slots__ = ()
    
    @abstractmethod
    def __call__(self, x):
        pass
    
class GaussianBasis(RBFTypes):
    '''
    '''
    __slots__ = ('__sigma')
    
    def __init__(self, sigma):
        '''
        '''
        self.__sigma = sigma
    
    def __call__(self, r):
        return np.exp(-(self.__sigma*r)**2)
    
class MultiquadricBasis(RBFTypes):
    '''
    '''
    __slots__ = ('__sigma')
    
    def __init__(self, sigma):
        '''
        '''
        self.__sigma = sigma
    
    def __call__(self, r):
        return np.sqrt(1 + (self.__sigma*r)**2.0)

class InverseQuadricBasis(RBFTypes):
    '''
    '''
    __slots__ = ('__sigma')
    
    def __init__(self, sigma):
        '''
        '''
        self.__sigma = sigma
    
    def __call__(self, r):
        return 1.0 / (1.0 + (self.__sigma*r)**2.0)

class InverseMultiquadricBasis(RBFTypes):
    '''
    '''
    __slots__ = ('__sigma')
    
    def __init__(self, sigma):
        '''
        '''
        self.__sigma = sigma
    
    def __call__(self, r):
        return 1.0 / np.sqrt(1.0 + (self.__sigma*r)**2.0)

class PolyharmonicBasis(RBFTypes):
    '''
    '''
    __slots__ = ('__k')
    
    def __init__(self, k):
        '''
        '''
        self.__k = k
    
    def __call__(self, r):
        return r**self.__k if self.__k % 2 else r**self.__k * np.log(r)
    
class ThinPlateBasis(RBFTypes):
    '''
    '''
    __slots__ = ('__k')
    
    def __init__(self, k):
        '''
        '''
        self.__k = k
    
    def __call__(self, r):
        return r**2.0 * np.log(r)
    
class CubicBasis(RBFTypes):
    '''
    '''
    __slots__ = ()
    
    def __call__(self, r):
        return r**3
    
class QuinticBasis(RBFTypes):
    '''
    '''
    __slots__ = ()
    
    def __call__(self, r):
        return r**5
    
#Factory functions for different basis functions
def polynomial(n):
    return PolynomialFunction(n)

def rbf(x, kernel=GaussianBasis(0.25)):
    return RadialBasisFunction.make_rbf_from_data(x, kernel)

def rbf_kmeans(x, kernel=GaussianBasis(0.25), ratio=0.05):
    return RadialBasisFunction.make_rbf_from_kmeans(x, kernel, ratio)

