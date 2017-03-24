'''
Created on Aug 9, 2011

@author: marcel
'''
import numpy as np
import scipy.special as sps

from abc import ABCMeta, abstractmethod

class Prior(object):
    '''
    @todo: - make mean() and variance() function an abstract property
    '''
    
    __metaclass__ = ABCMeta
    
    __slots__ = ()
    
    def plot(self, start, end, n=100):
        if start > end:
            raise ValueError('start must be greater than end.')
        
        import pylab as pl
        x = np.linspace(start, end, n)
        y = np.exp(self.log_pdf(x))
        pl.plot(x,y)
        
    def support(self, x):
        return True
        
    @abstractmethod
    def log_pdf(self, x):
        pass
    
    @abstractmethod
    def log_gradient(self, x):
        pass
    
    @abstractmethod
    def mean(self):
        pass
    
    @abstractmethod
    def variance(self):
        pass
    
class NormalPrior(Prior):        
    
    __slots__ = ('mu', 'sigma', '_const')
    
    def __init__(self, mu=0.0, sigma=1.0):
        if sigma < 0.0:
            raise ValueError('sigma must be positive.')
        
        self.mu = mu
        self.sigma = sigma
        self._const = np.log(1.0/(sigma*np.sqrt(2.0*np.pi)))
        
    def mean(self):
        return self.mu
    
    def variance(self):
        return self.sigma**2
    
    def log_pdf(self, x):
        '''
        @todo: -check this
        '''
        pdf = self._const - ((x-self.mu)/self.sigma)**2 / 2.0
        return pdf
    
    def log_gradient(self, x):
        grad = (self.mu-x)/self.sigma**2.0
        return grad
    
    def __str__( self ):
        """Returns a string representation of the distribution"""
        return "NormalPrior(mu={0}, sigma={1})".format(self.mu, self.sigma)

    
class LogNormalPrior(Prior):
    
    __slots__ = ('mu', 'sigma')
    
    def __init__(self, mu=0.0, sigma=1.0):
        if sigma < 0.0:
            raise ValueError('sigma must be positive.')
        
        self.mu = mu
        self.sigma = sigma
    
        
    def support(self, x):
        return x > 0.0
        
    def mean(self):
        return np.exp((self.mu+self.sigam**2)/2.0)
    
    def variance(self):
        s2 = self.sigma**2
        return (np.exp(s2)-1.0) * np.exp(2*self.mu+s2)
    
    def log_pdf(self, x):
        '''
        @todo: -check this
        '''
        if self.support(x) == False:
            return np.log(1e-300)
        
        mu = self.mu
        sigma = self.sigma
        
        ln_x = np.log(x)
        pdf = -ln_x - np.log(sigma) - 0.5*np.log(2*np.pi) - ((ln_x-mu)/sigma)**2/2.0
        return pdf
    
    def log_gradient(self, x):
        if self.support(x) == False:
            return 0
        
        mu = self.mu
        sigma = self.sigma
        s2 = sigma**2
        
        grad = -(np.log(x)+s2-mu)/(x*s2)
        return grad
    
    def __str__( self ):
        """Returns a string representation of the distribution"""
        return "LogNormalPrior(mu={0}, sigma={1})".format(self.mu, self.sigma)
    
    
class GammaPrior(Prior):
    '''
    '''
    
    __slots__ = ('k', 'theta')
    
    def __init__(self, k=1.0, theta=1.0):
        '''
        @todo: - check for the right default values
        '''
        if k < 0:
            raise ValueError('k must be positive')
        if theta < 0:
            raise ValueError('theta must be positive')
        
        self.k = k
        self.theta = theta
        
    def support(self, x):
        return x > 0.0
    
    def mean(self):
        return self.k*self.theta
    
    def variance(self):
        return self.k * self.theta**2
    
    def log_pdf(self, x):
        if self.support(x) == False:
            return np.log(1e-300)
        
        k = self.k
        theta = self.theta
        
        pdf = (k-1.0)*np.log(x) - (x/theta) - k*np.log(theta) - sps.gammaln(x)
        return pdf
    
    def log_gradient(self, x):
        if self.support(x) == False:
            return 0
        
        k = self.k
        theta = self.theta
        
        grad = (k-1.0)/x - 1.0/theta 
        return grad
    
    def __str__( self ):
        """Returns a string representation of the distribution"""
        return "GammaPrior(k={0}, theta={1})".format( self.k, self.theta )
