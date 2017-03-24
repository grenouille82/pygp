'''
Created on Mar 14, 2011

'''

import numpy as np
import numpy.linalg as lin
import scikits.optimization as sopt
import scipy.special as sps
import time

from upgeo.regression.linear import LinearRegressionModel, RidgeRegression
from upgeo.util.metric import tspe, mspe
from upgeo.util.stats import mvnpdf
from upgeo.regression.function import GaussianBasis
from numpy.lib.npyio import savetxt
from scipy.linalg.decomp_cholesky import cho_solve
from numpy.linalg.linalg import LinAlgError

class MapBayesRegression(LinearRegressionModel):
    '''
    @todo: - use evidence likelihood function to compute the parameters and the model likilihood
           - rename log_evidence in likelihood
    '''
    __slots__ = ('__alpha', 
                 '__beta', 
                 '__invA', 
                 '__A',
                 '__log_evidence',
                 '_weight_bias'
                 'X' #todo: remove after debugging
                 )
    
    def __init__(self, alpha=1.0, beta=1.0, basis_function=None, weight_bias=True):
        LinearRegressionModel.__init__(self, basis_function)
        if alpha < 0.0 or beta < 0.0:
            raise ValueError('alpha and beta must be non-negative')
        
        self.__alpha = alpha
        self.__beta = beta
        
        self._weight_bias = weight_bias
        
    def fit(self, X, y):
        '''
        '''
        X = self._preprocess_data(X)
        y = np.asarray(y)
        self.X = X
        
        if X.ndim != 2:
            raise TypeError('X must be two dimensional')
        if y.ndim != 1:
            raise TypeError('y must be one dimensional')
        
        [n,m] = X.shape
        X = np.c_[np.ones(n), X] 

        if self._weight_bias == False:
            D = np.diag(np.r_[0, np.ones(m)])
        else:
            m += 1
            D = np.diag(np.ones(m))
            
        A = self.__alpha*D + self.__beta*np.dot(X.T, X)
        invA = np.linalg.pinv(A) 
        w = self.__beta*np.dot(invA, np.dot(X.T, y))
         
        self._intercept = w[0]
        self._weights   = w[1:]
        self.__A = A
        self.__invA = invA
        self._is_init = True
        
        #calculate the log evidence p(y|alpha,beta) = int p(y|w,beta)*p(w|alpha) d_w
        self.__log_evidence = self.__estimate_evidence(X,y)

    def predict(self, X, ret_var=False):
        yhat = LinearRegressionModel.predict(self, X)
        if ret_var is True:
            X = self._preprocess_data(X)
            n = X.shape[0]
            X = np.c_[np.ones(n), X]
            #L = np.linalg.cholesky(self.__invA)
            #V = np.dot(L, X.T)
            #V = np.linalg.solve(L, X.T)
            #var = 1.0/self.__beta + np.sum(V*V,0)
            var = 1.0/self.__beta + np.diag(np.dot(np.dot(X, self.__invA), X.T))
            #var = 0.534 * np.ones(len(var))
            print 'var={0}'.format(var)
            print 'beta={0}'.format(self.__beta)
            return (yhat, var)
        return yhat
        
    def posterior(self, weights):
        '''
        Estimates and returns the posterior of the weights given the data p(w|y).
        The model must before fitted. 
        '''
        self._init_check()
        
        mean = np.r_[self.__intercept, self.__weights]
        sigma = np.linalg.inv(self.__precision) #todo: bug
        return mvnpdf(weights, mean, sigma)
    
    def _get_log_evidence(self):
        self._init_check()
        return self.__log_evidence
    
    log_evidence = property(fget=_get_log_evidence)
    
    log_likel = property(fget=_get_log_evidence)
    
    def _get_alpha(self):
        return self.__alpha
    
    alpha = property(fget=_get_alpha)
    
    def _get_beta(self):
        return self.__beta
    
    beta = property(fget=_get_beta)
    
    def __estimate_evidence(self, X, y):
        ''' Calculate the log evidence 
        ln p(y|alpha,beta) = ln int p(y|w,beta)*p(w|alpha) d_w
        @todo: - should have the same result as the EvidenceLikelihood class returns,
                 so the computation can be replaced
               - use cholesky decomposition for the determinant
        '''
        self._init_check()
        
        [n,m] = X.shape
        if self._weight_bias == False:
            m -= 1
        
        alpha = self.__alpha
        beta = self.__beta
        w = np.r_[self._intercept, self._weights]
        A = self.__A
        detA = np.linalg.det(A)
        yhat = np.dot(X, w)
        
        #print 'alpha={0}, beta={1}, n={2}, m={3}'.format(self.__alpha, self.__beta, n, m)
        
        err = self.__beta*tspe(y, yhat) + self.__alpha*np.dot(w,w)
        l = m*np.log(alpha) +  n*np.log(beta)
        l -= err + n*np.log(2.0*np.pi)
        if detA > 0:
            l -= np.log(detA)
            
        l /= 2.0
        return l
    
class RobustBayesRegression(LinearRegressionModel):
    
    __slots__ = ('_a0',     #hyperparameter in log space
                 '_b0',     #hyperparameter in log space
                 '_L0',      #precision matrix      
                 '_an',     #posterior hyperparameter in log space
                 '_bn',     #posterior hyperparameter in log space
                 '_mn',     #posterior mean
                 '_Ln',     #posterior precision matrix
                 '_log_likel', #model likelihood
                 'likel_fun', #likel_function
                 '_X',
                 '_y'
                 )
    
    def __init__(self, a0, b0, L0=None, basis_function=None):
        LinearRegressionModel.__init__(self, basis_function)
        
        self._a0 = a0
        self._b0 = b0
        self._L0 = L0
        
    def fit(self, X, y):
        '''
        '''
        X = self._preprocess_data(X)
        y = np.asarray(y)
        self._X = X
        self._y = y
        
        if X.ndim != 2:
            raise TypeError('X must be two dimensional')
        if y.ndim != 1:
            raise TypeError('y must be one dimensional')
        
        [n,m] = X.shape
        X = np.c_[np.ones(n), X]
        m += 1
        
        a0 = self._a0
        b0 = self._b0
        L0 = self._L0
        
        if L0 == None:
            L0 = np.eye(m)
            self._L0 = L0
        params = RobustBayesRegression.wrap(a0, b0, L0)
        
        likel_fun = RobustBayesRegression.LikelihoodFunction(X, y)
        likel = likel_fun(params)
        
        self._log_likel = likel
        self.likel_fun = likel_fun
        
        self._Ln = likel_fun._Ln
        self._mn = likel_fun._mn
        self._an = likel_fun._an
        self._bn = likel_fun._bn
        
        self._intercept = self._mn[0]
        self._weights   = self._mn[1:]
        self._is_init = True
    
    def predict(self, X, ret_var=False):
        yhat = LinearRegressionModel.predict(self, X)
        if ret_var is True:
            X = self._preprocess_data(X)
            n = X.shape[0]
            X = np.c_[np.ones(n), X]
            L = self._Ln
            V = np.dot(L, X.T)
            var = 1.0/self._bn + np.sum(V*V,0)
            #var = 1.0/self.__beta + np.diag(np.dot(np.dot(X, self.__invA), X.T))
            return (yhat, var)
        return yhat

    
    def refit(self, X, y, params):
        a,b,L = RobustBayesRegression.unwrap(params)
        
        n = X.shape[0]
        X = np.c_[np.ones(n), X]
        
        self._a0 = a
        self._b0 = b
        self._L0 = L
        
        likel_fun = RobustBayesRegression.LikelihoodFunction(X, y)
        likel = likel_fun(params)

        self._log_likel = likel
        self.likel_fun = likel_fun
        
        self._Ln = likel_fun._Ln
        self._mn = likel_fun._mn
        self._an = likel_fun._an
        self._bn = likel_fun._bn
        
        self._intercept = self._mn[0]
        self._weights   = self._mn[1:]
        self._is_init = True
        
            
    def posterior(self, X):
        pass
        
    def _get_log_likel(self):
        self._init_check()
        return self._log_likel
    
    log_likel = property(fget=_get_log_likel)
    
    def _get_hyperparams(self):
        return self._a0, self._b0, self._L0
    
    hyperparams = property(fget=_get_hyperparams)
    
    
    class LikelihoodFunction(object):
        
        __slots__ = ('_XX',
                     '_yy',
                     '_Xy',
                     
                     '_n',
            
                     '_old_params',
                     
                     '_mn',
                     '_Ln',
                     '_an',
                     '_bn'
                     )
        
        def __init__(self, X, y):
            XX = np.dot(X.T,X)
            yy = np.dot(y,y)
            Xy = np.dot(X.T,y)
            
            self._n = len(X)
            
            self._XX = XX
            self._yy = yy
            self._Xy = Xy
            
            self._old_params = None
        
        def __call__(self, params):
            '''
            @todo: - error handling
            '''
            n = self._n
            #print 'params'
            #print params
            
            old_params = self._old_params
            if old_params == None or np.any((np.not_equal(old_params, params))):
                a,b,L = RobustBayesRegression.unwrap(params)
                a = np.exp(a)
                b = np.exp(b)
                L = np.tril(L)
                
                
                #print 'a={0}, b={1}, L={2}'.format(a,b,L)
                
                #check if L is positive semidefinit
                if np.any(np.diag(L) < 0):
                #if np.any(L < 0):
                    return -1e300 #todo eventually return inf 
                
                XX = self._XX
                yy = self._yy
                Xy = self._Xy
                
                L += 1e-6*np.eye(len(XX)) #add some jitter
                V = np.dot(L, L.T)
                
                #posteriors
                Vn = V + XX
                #print 'V'
                #print V
                #print 'Vn'
                #print Vn
                #try:
                #    Ln = np.linalg.cholesky(Vn + 1e-6*np.eye(len(XX))) #add some jitter
                #except LinAlgError:
                #    print 'V'
                #    print V
                #    print 'Vn'
                #    print Vn
                Ln = np.linalg.cholesky(Vn + 1e-6*np.eye(len(XX))) #add some jitter
                mn = cho_solve((Ln, 1), Xy)
                an = a + n/2.0
                bn = b + (yy - np.sum(np.dot(mn, Ln)**2))/2.0
                
                self._Ln = Ln
                self._mn = mn
                self._an = an
                self._bn = bn
                
                self._old_params = params
            else:
                a,b,L = RobustBayesRegression.unwrap(old_params)
                a = np.exp(a)
                b = np.exp(b)
                L = np.tril(L)
                L += 1e-6*np.eye(len(self._XX)) #add some jitter
                
                #posteriors are already computed
                Ln = self._Ln
                mn = self._mn
                an = self._an
                bn = self._bn
            
            #compute marginal likelihood
            A = np.sum(np.log(np.diag(L))) - np.sum(np.log(np.diag(Ln)))
            B = a*np.log(b) - an*np.log(bn)
            C = sps.gammaln(an) - sps.gammaln(a)
            
            likel = A + B + C - n/2.0*np.log(2.0*np.pi)
            return likel
        
        def gradient(self, params):
            '''
            @todo: - error handling
            '''
            
            old_params = self._old_params
            if old_params == None or np.any((np.not_equal(old_params, params))):
                a,b,L = RobustBayesRegression.unwrap(params)
                a = np.exp(a)
                b = np.exp(b)
                L = np.tril(L)
                
                #check if L is positive semidefinit
                if np.any(np.diag(L) < 0):
                #if np.any(L < 0):
                    return 0 #todo eventually return inf 

                
                XX = self._XX
                yy = self._yy
                Xy = self._Xy
                
                n = self._n
                
                L += 1e-6*np.eye(len(self._XX)) #add some jitter
                V = np.dot(L, L.T)
                
                #posteriors
                Vn = V + XX
                Ln = np.linalg.cholesky(Vn)
                mn = cho_solve((Ln, 1), Xy)
                an = a + n/2.0
                bn = b + (yy - np.sum(np.dot(mn, Ln)**2))/2.0
                
                self._Ln = Ln
                self._mn = mn
                self._an = an
                self._bn = bn
                
                self._old_params = params
            else:
                a,b,L = RobustBayesRegression.unwrap(old_params)
                a = np.exp(a)
                b = np.exp(b)
                L = np.tril(L)
                L += 1e-6*np.eye(len(self._XX)) #add some jitter
                
                #posteriors are already computed
                Ln = self._Ln
                mn = self._mn
                an = self._an
                bn = self._bn

            #compute gradient for hyperparameter a
            B_prime = np.log(b) - np.log(bn)
            C_prime = sps.digamma(an) - sps.digamma(a)
            grad_a = (B_prime+C_prime) * a
            
            #compute gradient for hyperparameter b
            B_prime = a/b - an/bn
            grad_b = B_prime * b
            
            #computer gradient for the lower matrix L of the precision matrix
            #Inverses
            Xy = self._Xy
            XyyX = np.outer(Xy,Xy)
            H1 = cho_solve((Ln, 1), XyyX)
            H2 = cho_solve((Ln, 1), L)
            H3 = cho_solve((L, 1), L)

            A_prime = H3 - H2
            B_prime = (a/b - an/bn) * np.dot(H1,H2)  
            
            grad_L = np.tril(A_prime+B_prime)
            
            grad = RobustBayesRegression.wrap(grad_a, grad_b, grad_L)
            return grad
        
    @staticmethod
    def wrap(a, b, L):
        params = np.r_[a, b, L.ravel()]
        return params
    
    @staticmethod
    def unwrap(params):
        n = len(params)
        m = np.sqrt(n-2)
        
        a = params[0]
        b = params[1]
        
        L = np.reshape(params[2:n], (m,m))
        
        return (a,b,L)
    
class GaussianBayesRegression(LinearRegressionModel):
    '''
    @todo: - use evidence likelihood function to compute the parameters and the model likilihood
           - rename log_evidence in likelihood
    '''
    __slots__ = ('__alpha', 
                 '__beta', 
                 '__invA', 
                 '__detA',
                 '__A',
                 '__log_evidence',
                 '_weight_bias'
                 'X' #todo: remove after debugging
                 )
    
    def __init__(self, alpha=1.0, beta=1.0, basis_function=None, weight_bias=False):
        LinearRegressionModel.__init__(self, basis_function)
        if alpha < 0.0 or beta < 0.0:
            raise ValueError('alpha and beta must be non-negative')
        
        self.__alpha = alpha
        self.__beta = beta
        
        self._weight_bias = weight_bias
        
    def fit(self, X, y):
        '''
        '''
        X = self._preprocess_data(X)
        y = np.asarray(y)
        self.X = X
        
        if X.ndim != 2:
            raise TypeError('X must be two dimensional')
        if y.ndim != 1:
            raise TypeError('y must be one dimensional')
        
        [n,m] = X.shape
        X = np.c_[np.ones(n), X] 

        if self._weight_bias == False:
            D = np.diag(np.r_[0, np.ones(m)])
        else:
            m += 1
            D = np.diag(np.ones(m))
        
        sigma = np.sqrt(1.0/self.__beta)
        [w, S, detS] = self._solve_normal_equations(X, y, D, sigma)
         
        self._intercept = w[0]
        self._weights   = w[1:]

        self.__invA = S
        self.__detA = detS
        self._is_init = True
        
        #calculate the log evidence p(y|alpha,beta) = int p(y|w,beta)*p(w|alpha) d_w
        self.__log_evidence = self.__estimate_evidence(X,y)
        
    def posterior(self, weights):
        '''
        Estimates and returns the posterior of the weights given the data p(w|y).
        The model must before fitted. 
        '''
        self._init_check()
        
        mean = np.r_[self.__intercept, self.__weights]
        sigma = np.linalg.inv(self.__precision) #todo: bug
        return mvnpdf(weights, mean, sigma)
    
    def _solve_normal_equations(self, X, y, D, sigma):
        if np.all(np.diag(D) == 0):
            Q,R = np.linalg.qr(X)
            w = None #OLS
            invR = np.linalg.inv(R)
            S = sigma**2.0 * np.dot(invR, invR.T)
            detS = np.sum(np.log(np.diag(R)))
        else:
            n = len(D)
            L = np.linalg.cholesky(D)
            Xtilde = np.r_[X/sigma, L]
            ytilde = np.r_[y/sigma, np.zeros(n)]
            Q,R = np.linalg.qr(Xtilde)
            w = None #OLS
            invR = np.linalg.inv(R)
            S = np.dot(invR, invR.T)
            detS = np.linalg.inv(R)
            
        return w, S, detS
        
    def _get_log_evidence(self):
        self._init_check()
        return self.__log_evidence
    
    log_evidence = property(fget=_get_log_evidence)
    
    def _get_alpha(self):
        return self.__alpha
    
    alpha = property(fget=_get_alpha)
    
    def _get_beta(self):
        return self.__beta
    
    beta = property(fget=_get_beta)
    
    def __estimate_evidence(self, X, y):
        ''' Calculate the log evidence 
        ln p(y|alpha,beta) = ln int p(y|w,beta)*p(w|alpha) d_w
        @todo: - should have the same result as the EvidenceLikelihood class returns,
                 so the computation can be replaced
        
        '''
        self._init_check()
        
        [n,m] = X.shape
        if self._weight_bias == False:
            m -= 1
        
        alpha = self.__alpha
        beta = self.__beta
        w = np.r_[self._intercept, self._weights]
        A = self.__A
        detA = self.__detA*2.0
        yhat = np.dot(X, w)
        
        #print 'alpha={0}, beta={1}, n={2}, m={3}'.format(self.__alpha, self.__beta, n, m)
        
        err = self.__beta*tspe(y, yhat) + self.__alpha*np.dot(w,w)
        l = m*np.log(alpha) +  n*np.log(beta)
        #check the last of the likelihood if it truth (last term: n or m)
        l -= err + detA + n*np.log(2.0*np.pi)
        l /= 2.0 
        return l
    
class FastBayesRegression(LinearRegressionModel):
    '''
    @todo: - use evidence likelihood function to compute the parameters and the model likilihood
           - rename log_evidence in likelihood
    '''
    __slots__ = ('__alpha', 
                 '__beta', 
                 '__invA', 
                 '__detA'
                 '__A',
                 '__log_evidence',
                 '_weight_bias'
                 'likel_fun'
                 'X' #todo: remove after debugging
                 )
    
    def __init__(self, alpha=1.0, beta=1.0, basis_function=None, weight_bias=False):
        LinearRegressionModel.__init__(self, basis_function)
        if alpha < 0.0 or beta < 0.0:
            raise ValueError('alpha and beta must be non-negative')
        
        self.__alpha = alpha
        self.__beta = beta
        
        self._weight_bias = weight_bias
        
    def refit(self, X, y, alpha=None, beta=None):
        if alpha != None:
            self.__alpha = alpha
        if beta != None:
            self.__beta = beta
        self.fit(X,y)
        
    def fit(self, X, y):
        '''
        '''
        X = self._preprocess_data(X)
        y = np.asarray(y)
        self.X = X
        
        if X.ndim != 2:
            raise TypeError('X must be two dimensional')
        if y.ndim != 1:
            raise TypeError('y must be one dimensional')
        
        [n,m] = X.shape
        X = np.c_[np.ones(n), X]
        m += 1
        
        if n > m:
            if self._weight_bias == False:
                m -= 1
                D = np.diag(np.r_[0, np.ones(m)]*self.__alpha)
            else:
                D = np.diag(np.ones(m)*self.__alpha)
            
            A = D + self.__beta*np.dot(X.T, X)
            L = np.linalg.cholesky(A)
            U = np.linalg.inv(L)
            invA = np.dot(U.T, U)
            detA = np.sum(np.log(np.diag(L))) * 2.0
        else:
            A = np.eye(n)/self.__beta + np.dot(X, X.T)/self.__alpha
            L = np.linalg.cholesky(A)
            U = np.linalg.inv(L)
            G = np.dot(np.dot(np.dot(X.T, U.T), U), X) / self.__alpha / self.__alpha
            Z = np.eye(m)/self.__alpha
            Z[0,0] = 0
            invA = np.eye(m)/self.__alpha - G
            detA = np.sum(np.log(np.diag(L))) * 2.0 + m*np.log(self.__alpha) + n*np.log(self.__beta)
                 
        w = self.__beta*np.dot(invA, np.dot(X.T, y))
         
        self._intercept = w[0]
        self._weights   = w[1:]
        self.__A = A
        self.__invA = invA
        self.__detA = detA
        self._is_init = True
        
        #calculate the log evidence p(y|alpha,beta) = int p(y|w,beta)*p(w|alpha) d_w
        self.__log_evidence = self.__estimate_evidence(X,y)
        self.likel_fun = BayesEvidenceLikelihood(X,y)
        
    def predict(self, X, ret_var=False):
        yhat = LinearRegressionModel.predict(self, X)
        if ret_var is True:
            X = self._preprocess_data(X)
            n = X.shape[0]
            X = np.c_[np.ones(n), X]
            #L = np.linalg.cholesky(self.__invA)
            #V = np.dot(L, X.T)
            #var = 1.0/self.__beta + np.sum(V*V,0)
            var = 1.0/self.__beta + np.diag(np.dot(np.dot(X, self.__invA), X.T))
            #var = 0.534 * np.ones(len(var))
            print 'var={0}'.format(var)
            print 'beta={0}'.format(self.__beta)
            return (yhat, var)
        return yhat
        
        
    def posterior(self, weights):
        '''
        Estimates and returns the posterior of the weights given the data p(w|y).
        The model must before fitted. 
        '''
        self._init_check()
        
        mean = np.r_[self.__intercept, self.__weights]
        sigma = np.linalg.pinv(self.__A) #todo: bug!?
        return mvnpdf(weights, mean, sigma)
    
    def _get_log_evidence(self):
        self._init_check()
        return self.__log_evidence
    
    log_evidence = property(fget=_get_log_evidence)
    
    log_likel = property(fget=_get_log_evidence)
    
    def _get_alpha(self):
        return self.__alpha
    
    alpha = property(fget=_get_alpha)
    
    def _get_beta(self):
        return self.__beta
    
    beta = property(fget=_get_beta)
    
    def _get_hyperparams(self):
        return np.asarray([self.__alpha, self.__beta])
    
    hyperparams = property(fget=_get_hyperparams)
    
    def __estimate_evidence(self, X, y):
        ''' Calculate the log evidence 
        ln p(y|alpha,beta) = ln int p(y|w,beta)*p(w|alpha) d_w
        @todo: - should have the same result as the EvidenceLikelihood class returns,
                 so the computation can be replaced
        
        '''
        self._init_check()
        
        [n,m] = X.shape
        if self._weight_bias == False:
            m -= 1
        
        alpha = self.__alpha
        beta = self.__beta
        w = np.r_[self._intercept, self._weights]
        detA = self.__detA
        yhat = np.dot(X, w)
        
        #print 'alpha={0}, beta={1}, n={2}, m={3}'.format(self.__alpha, self.__beta, n, m)
        
        err = self.__beta*tspe(y, yhat) + self.__alpha*np.dot(w,w)
        l = m*np.log(alpha) +  n*np.log(beta)
        #check the last of the likelihood if it truth (last term: n or m)
        l -= err + detA + n*np.log(2.0*np.pi)
        l /= 2.0 
        return l

class BayesEvidenceLikelihood(object):
    '''
    @todo: - the result must be identical to the evidence likelihood of
             the map regression model
    '''
    
    __slots__ = ('_X', 
                 '_y',
                 '_XX',
                 '_Xy',
                 '_n',
                 '_m',
                 '_wbias',  #flag checks whether the bias term should be weighted by prior
                 '_D',      
                 'A',       #precision matrix
                 'invA',    #covariance matrix
                 'w',       #the mean vector or weights
                 
                 )
        
        
    def __init__(self, X, y, weight_bias=True):
        '''
        
        '''
        X = np.asarray(X)
        y = np.asarray(y)
        
        self._n, self._m = X.shape
        self._wbias = weight_bias
        
        self._X = np.c_[np.ones(self._n), X]
        self._y = y
        
        self._XX = np.dot(self._X.T, self._X)
        self._Xy = np.dot(self._X.T, self._y)
        
        if weight_bias == False:
            self._D = np.diag(np.r_[0, np.ones(self._m)])
            #self._m +=1
        else:
            self._m +=1
            self._D = np.diag(np.ones(self._m))
            
        
    def __call__(self, params):
        '''
        @todo: - use cholesky decomposition for computing the determinant of the matrix
                 (how should be handled the sign)
                 
        ''' 
        #print 'likel_params'
        #print params
        alpha = np.exp(params[0])
        beta = np.exp(params[1])
        
        m = self._m
        n = self._n
        D = self._D
        X = self._X
        y = self._y
        XX = self._XX
        Xy = self._Xy
        
        A = alpha*D + beta*XX
        invA = lin.pinv(A) 
        w = beta*np.dot(invA, Xy)
        yhat = np.dot(X, w)
        
        err = beta*tspe(y, yhat) + alpha*np.dot(w,w)
        _, ln_detA = np.linalg.slogdet(A)
        
        l = m*np.log(alpha) + n*np.log(beta)
        l -= err + self._n*np.log(2.0*np.pi) + ln_detA 
        l /= 2.0
        
        self.w = w
        self.A = A
        self.invA = invA
        
        return l
    
    def gradient(self, params):
        
        #print 'grad_params'
        #print params
        alpha = np.exp(params[0])
        beta = np.exp(params[1])
          
        D = self._D
        XX = self._XX
        Xy = self._Xy
        
        A = alpha*D + beta*XX
        invA = lin.pinv(A) 
        w = beta*np.dot(invA, Xy)
        
        self.w = w
        self.A = A
        self.invA = invA
        
        grad_a = self._gradient_alpha(alpha, beta) * alpha
        grad_b = self._gradient_beta(alpha, beta) * beta
        
        return np.array([grad_a, grad_b])
        
    def _gradient_alpha(self, alpha, beta):

        m = self._m
        X = self._X
        y = self._y
        
        w = self.w
        invA = self.invA
        
        Xw = np.dot(X,w)
        ww = np.dot(w,w)
        
        E = ww - 2.0*alpha*np.dot(np.dot(w, invA), w)
        D = 2.0*beta*np.dot(np.dot(X.T, (y-Xw)), np.dot(invA, w))
        
        grad = (m/alpha - np.trace(invA) - (E+D)) / 2.0
        return grad
    
    def _gradient_beta(self, alpha, beta):
        n = self._n
        D = self._D
        X = self._X
        y = self._y
        XX = self._XX
        Xy = self._Xy
        
        w = self.w
        invA = self.invA
        
        Xw = np.dot(X,w)
        w_prime = np.dot(np.dot(invA, D-np.dot(XX, invA)), Xy)
        z = y-Xw
        
        E = 2.0*alpha*np.dot(w, w_prime)
        D = np.dot(z,z) - 2.0*beta*np.dot(np.dot(X.T, z), w_prime)
        
        grad = (n/beta - np.trace(np.dot(invA, XX)) - (E+D)) / 2.0
        return grad
    
        
class EMBayesRegression(LinearRegressionModel):
    '''
    @todo: - handling zero values for alpha and beta
           - the computation of not weighting the bias term have a bug
    '''
    __slots__ = ('_alpha',
                 '_alpha0',
                 '_beta',
                 '_beta0',
                 '_weight_bias'
                 '_likel_tol',
                 '_max_it',
                 '_reg_model'
                 )
    
    def __init__(self, alpha0=1.0, beta0=1.0, basis_function=None, weight_bias=True,
                 likel_tol=10e-16, max_it=100):
        LinearRegressionModel.__init__(self, basis_function)
        
        if alpha0 < 0.0 or beta0 < 0.0:
            raise ValueError('alpha0 and beta0 must be non-negative')
        if likel_tol < 0.0:
            raise ValueError('likel tolerance must be non-negative')
        if max_it < 1:
            raise ValueError('max number of iterations must be greater than 0')
        
        self._alpha0 = alpha0
        self._beta0 = beta0
        
        self._weight_bias = weight_bias
        
        self._likel_tol = likel_tol
        self._max_it = max_it
    
    def fit(self, X, y):
        '''
        '''
        X = self._preprocess_data(X)
        y = np.asarray(y) 
        
        if X.ndim != 2:
            raise TypeError('X must be two dimensional')
        if y.ndim != 1:
            raise TypeError('y must be one dimensional')
        
        [m,d] = X.shape
        Xb = np.c_[np.ones(m), X]
        #d += 1
        if self._weight_bias:
            d += 1
        
        likel_fun = EvidenceLikelihood(X, y, self._weight_bias)
        
        alpha = self._alpha0
        beta = self._beta0
        
        #print 'em_hyperparams={0}'.format(self.hyperparams)
           
        i = 0
        while i < self._max_it: 
            likel_new = likel_fun(alpha, beta)
            
            print 'likel={0}, alpha={1}, beta={2}'.format(likel_new, alpha, beta)
            
            ##########################
            #E-Step:
            ##########################
            invA = likel_fun.invA
            #estimate the expectation E[w]
            w = likel_fun.w
            #estimate the expectation E[w'w]
            w_prime_w = np.dot(w,w)+np.trace(invA)
            #estimate the expectation E[ww']
            ww_prime = invA + np.outer(w,w)
            
            ##########################
            #M-Step
            ##########################
            #maximize alpha = d / E[w'w]
            #todo: include the bias dimension or not
            alpha_new = d/w_prime_w 
                
            #maximize beta = m / (y'y - 2*y'*X*E[w] + Tr(X*E[ww']*X'))
            denom = np.dot(y,y) 
            denom -= 2*np.dot(y, np.dot(Xb, w))
            #denom -= 2*np.dot(np.dot(y, Xb), w)
            denom += np.trace(np.dot(Xb.T, np.dot(Xb, ww_prime)))
    
            beta_new = m / denom
            
            likel_old = likel_new
            likel_new = likel_fun(alpha_new, beta_new)
            
            i += 1
            
            #print likel_old
            #print likel_new
                
            if likel_new-likel_old < self._likel_tol:
                break
            
            alpha = alpha_new
            beta = beta_new
            
        self._alpha = alpha
        self._beta  = beta
        
        self._reg_model = MapBayesRegression(self._alpha, self._beta)
        self._reg_model.fit(X,y)
        self._intercept = self._reg_model.intercept
        self._weights = self._reg_model.weights
        self._is_init = True
        
    def predict(self, X, ret_var=False):
        return self._reg_model.predict(X, ret_var)
        
    def _get_log_evidence(self):
        self._init_check()
        return self._reg_model.log_evidence
    
    log_evidence = property(fget=_get_log_evidence)
    
    log_likel = property(fget=_get_log_evidence)
    
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        return np.asarray([self._alpha0, self._beta0])
    
    def _set_hyperparams(self, params):
        '''
        '''
        self._alpha0 = params[0]
        self._beta0 = params[1] 
    
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)
    
    def _get_alpha(self):
        return self._alpha
    
    alpha = property(fget=_get_alpha)
    
    def _get_beta(self):
        return self._beta
    
    beta = property(fget=_get_beta)
    
class EBChenRegression(LinearRegressionModel):
    '''
    @todo: - - the computation of not weighting the bias term have a bug
    '''
    __slots__ = ('_alpha',
                 '_alpha0',
                 '_beta',
                 '_beta0',
                 '_likel_tol',
                 '_max_it',
                 '_weight_bias',
                 '_log_evidence',
                 '_reg_model'
                 )
    
    def __init__(self, alpha0=0.1, beta0=1.0, basis_function=None, weight_bias=True,
                 likel_tol=1e-2, max_it=100):
        LinearRegressionModel.__init__(self, basis_function)
        
        if alpha0 < 0.0 or beta0 < 0.0:
            raise ValueError('alpha0 and beta0 must be non-negative')
        if likel_tol < 0.0:
            raise ValueError('likel tolerance must be non-negative')
        if max_it < 1:
            raise ValueError('max number of iterations must be greater than 0')
        
        self._alpha0 = alpha0
        self._beta0 = beta0
        
        self._likel_tol = likel_tol
        self._max_it = max_it
        
        self._weight_bias = weight_bias
    
    
    def fit(self, X, y):
        '''
        @todo: - cleanup
        '''
        X = self._preprocess_data(X)
        y = np.asarray(y) 
        
        if X.ndim != 2:
            raise TypeError('X must be two dimensional')
        if y.ndim != 1:
            raise TypeError('y must be one dimensional')
        
        [n,m] = X.shape
        Xb = np.c_[np.ones(n), X]
        
        if self._weight_bias: 
            m += 1
        likel_fun = EvidenceLikelihood(X, y, self._weight_bias)
        
        alpha = self._alpha0
        beta = self._beta0
           
        likel = likel_fun(alpha, beta)
        i = 0
        while i < self._max_it: 
            #likel_fun(alpha, beta)
            #likel = likel_fun(alpha, beta)
            
            invA = likel_fun.invA
            w = likel_fun.w
            yhat = np.dot(Xb, w)
            
            #print 'likel={0}, alpha={1}, beta={2}, w={3}, step={4}'.format(likel, alpha, beta, w, i)
            
            gamma = m - alpha*np.trace(invA)
            beta_new = (n-gamma) / tspe(y, yhat)
            
            likel_old = likel
            likel = likel_fun(alpha, beta_new)
            
            #print 'likel={0}, alpha={1}, beta={2}, w={3}'.format(likel, alpha, beta_new, w)
            
            if np.abs(likel-likel_old) < self._likel_tol:
                break
            
            beta = beta_new
            alpha = gamma / np.dot(w,w)
            likel = likel_fun(alpha,beta)
            i += 1
            
        self._alpha = alpha
        self._beta  = beta
        #print 'alpha={0}, beta={1}'.format(alpha, beta)
        
        reg_model = FastBayesRegression(self._alpha, self._beta, 
                                        weight_bias=self._weight_bias)
        reg_model.fit(X,y)

        self._reg_model = reg_model
        
        self._intercept = reg_model.intercept
        self._weights = reg_model.weights
        self._log_evidence = reg_model.log_evidence
        
        self._is_init = True

    def predict(self, X, ret_var=False):
        return self._reg_model.predict(X, ret_var)


    def _get_log_evidence(self):
        self._init_check()
        return self._log_evidence
    
    log_evidence = property(fget=_get_log_evidence)
    
    def _get_alpha(self):
        self._init_check()
        return self._alpha
    
    alpha = property(fget=_get_alpha)
    
    def _get_beta(self):
        self._init_check()
        return self._beta
    
    beta = property(fget=_get_beta)

    
class MLABayesRegression(LinearRegressionModel):
    '''
    '''
    __slots__ = ('_reg_model',      #regression model with optimized parameters
                 '_alpha0'          #starting value for the precision parameter
                 '_alpha'           #MAP estimate of the weight precision parameter
                 '_beta'            #model variance is fixed to one
                 '_line_search'     #line search algorithm
                 '_stop_criterion'  #
                 '_opt_method'      #optimization method
                 )
    
    def __init__(self, alpha=1, basis_function=None, opt_method=sopt.step.GradientStep(), 
                 line_search=None, 
                 stop_criterion=sopt.criterion.criterion(ftol=10e-4, iterations_max=100)):
        LinearRegressionModel.__init__(self, basis_function)
        
        self._alpha0 = alpha
        self._beta = 1
        self._basis_function = basis_function
        
        self._opt_method = opt_method
        self._line_search = line_search
        self._stop_criterion = stop_criterion
        
    
    def fit(self, X, y):
        '''
        '''
        X = self._preprocess_data(X)
        y = np.asarray(y) 
        
        if X.ndim != 2:
            raise TypeError('X must be two dimensional')
        if y.ndim != 1:
            raise TypeError('y must be one dimensional')
        
        m = X.shape[0]
        X = np.c_[np.ones(m), X]
        
        likel_fun = AlphaEvidenceLikelihood(X, y, 1.0/self._beta)
        optimizer = sopt.optimizer.StandardOptimizer(function=likel_fun, 
                                                     step=self._opt_method,
                                                     line_search=self._line_search, 
                                                     criterion=self._stop_criterion, 
                                                     x0=self._alpha0)
        opt_point = optimizer.optimize()
        
        self._alpha = opt_point[0]
        self._reg_model = MapBayesRegression(self._alpha, self._beta, self._basis_function, )
        self._reg_model.fit(X,y)
        self._intercept = self._reg_model.intercept
        self._weights = self.weights
        
        self._is_init = True
        
    def _get_alpha(self):
        self._init_check()
        return self._alpha
    
    alpha = property(fget=_get_alpha)
    
    def _get_beta(self):
        return self._alpha
    
    beta = property(fget=_get_beta)

    def _get_log_evidence(self):
        self._init_check()
        return self._reg_model.log_evidence
    
    log_evidence = property(fget=_get_log_evidence)

class ExpectedEvidenceLikelihood(object):
    '''
    '''
    __slots__ = ('_X', 
                 '_y',
                 '_n',
                 '_d',
                 '_bias',
                 '_D',
                 'A',       #precision matrix
                 'invA',    #covariance matrix
                 'w',       #the mean vector or weights
                 'w_prime_w',
                 'ww_prime'
                 )
    
    def __init__(self, X, y, bias=True):
        '''
        '''
        self._X = X
        self._y = y
        self._n, self._d = X.shape
        self._bias = bias
        
        if bias == True:
            self._D = np.diag(np.r_[0, np.ones(self._d-1)])
            self._d += 1
        else:
            self._D = np.diag(np.ones(self._d))
        #self._d +=1
        
    def __call__(self, alpha, beta):
        '''
        ''' 
        #print 'alpha={0}, beta={1}, n={2}, m={3}'.format(alpha, beta, self._n, self._d)
        
        A = alpha*self._D + beta*np.dot(self._X.T, self._X)
        invA = lin.pinv(A) 
        w = beta*np.dot(np.dot(invA, self._X.T), self._y)
        
        #estimate the expectation E[w'w]
        w_prime_w = np.dot(w,w)+np.trace(invA)
        #estimate the expectation E[ww']
        ww_prime = invA + np.outer(w,w)
        
        #y'y - 2*y'*X*E[w] + Tr(X*E[ww']*X')
        expect_term = np.dot(self._y, self._y)
        expect_term -= 2.0*np.dot(self._y, np.dot(self._X, w)) 
        expect_term += np.trace(np.dot(self._X.T, np.dot(self._X, ww_prime)))
        
        l = self._d/2.0*np.log(alpha/2*np.pi) + self._n/2.0*np.log(beta/2*np.pi)
        l -= alpha/2.0*w_prime_w
        l -= beta/2.0*expect_term
        
        self.w = w
        self.A = A
        self.invA = invA
        self.w_prime_w = w_prime_w
        self.ww_prime = ww_prime
        
        return l  

    
class EvidenceLikelihood(object):
    '''
    @todo: - the result must be identical to the evidence likelihood of
             the map regression model
    '''
    
    __slots__ = ('_X', 
                 '_y',
                 '_XX',
                 '_Xy',
                 '_n',
                 '_m',
                 '_wbias',  #flag checks whether the bias term should be weighted by prior
                 '_D',      
                 'A',       #precision matrix
                 'invA',    #covariance matrix
                 'w'        #the mean vector or weights
                 )
        
        
    def __init__(self, X, y, weight_bias=True):
        '''
        
        '''
        X = np.asarray(X)
        y = np.asarray(y)
        
        self._n, self._m = X.shape
        self._wbias = weight_bias
        
        self._X = np.c_[np.ones(self._n), X]
        self._y = y
        
        self._XX = np.dot(self._X.T, self._X)
        self._Xy = np.dot(self._X.T, self._y)
        
        if weight_bias == False:
            self._D = np.diag(np.r_[0, np.ones(self._m)])
            #self._m +=1
        else:
            self._m +=1
            self._D = np.diag(np.ones(self._m))
            
        
    def __call__(self, alpha, beta):
        '''
        @todo: - use cholesky decomposition for computing the determinant of the matrix
                 (how should be handled the sign)
        ''' 
        #print 'alpha={0}, beta={1}, n={2}, m={3}'.format(alpha, beta, self._n, self._d)
        
        A = alpha*self._D + beta*self._XX
        invA = lin.pinv(A) 
        w = beta*np.dot(invA, self._Xy)
        
        yhat = np.dot(self._X, w)
        
        err = beta*tspe(self._y, yhat) + alpha*np.dot(w,w)
        _, log_detA = np.linalg.slogdet(A)
        
        l = self._m*np.log(alpha) +  self._n*np.log(beta)
        l -= err + self._n*np.log(2.0*np.pi) + log_detA 
        l /= 2.0
        
        self.w = w
        self.A = A
        self.invA = invA
        
        return l          

class SlowEvidenceLikelihood(object):
    
    __slots__ = ('X',
                 'y'
                 )        

    def __call__(self, alpha, beta):
        pass
        

class AlphaEvidenceLikelihood(object):
    '''
    '''
    
    __slots__ = ('_X',      #covariate matrix
                 '_XX', 
                 '_y',      #target vector
                 '_yy',
                 '_sigma',  #variance of the model
                 '_n'
                 )
    
    def __init__(self, X, y, sigma):
        '''
        '''
        self._X = np.asarray(X)
        self._XX = np.dot(self._X, self._X.T)
        self._y = np.asarray(y)
        self._yy = np.outer(self._y, self._y)
        self._n = self._X.shape[0]
        self._sigma = sigma
    
    def __call__(self, alpha):
        '''
        @todo: - maybe use stats.mvnlnpdf function for the computation
                 of the log likelihood
        '''
        D = np.diag(self._sigma*np.ones(self._n))
        A = D + alpha*self._XX
        invA = lin.pinv(A)
        
        likel = np.log(lin.det(A)) + np.dot(np.dot(self._y, invA), self._y)
        likel *= -0.5
        return likel
    
    def gradient(self, alpha):
        '''
        '''
        D = np.diag(self._sigma*np.ones(self._n))
        A = D + alpha*self._XX
        invA = lin.pinv(A)
        
        #-0.5*[Tr(A^-1*X*X') - Tr(A^-1*y*y'*A^-1*X*X')]
        grad = np.trace(np.dot(np.dot(np.dot(invA, self._yy), invA), self._XX))
        grad -= np.trace(np.dot(invA, self._XX))
        grad *= 0.5
          
        return grad
        #reg_model.fit(X, y)