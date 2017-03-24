'''
Created on Jul 1, 2013

@author: marcel
'''
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from upgeo.regression.linear import LinearRegressionModel
from upgeo.util.stats import mvnpdf

class LinearLRTORegression(LinearRegressionModel):
    '''
    Wrapper by using basic GP in a MTL setting by pooling the training data of the
    primary task and background tasks.
    '''
    __slots__ = ('_Xp',
                 '_yp',
                 '_Xs',
                 '_ys',
                 '_itasks',
                 
                 '_init_pi',
                 '_multi_bgr',
                 
                 '_wp',
                 '_ws',
                 '_betap',
                 '_betas',
                 '_pi'
                 )
            
    
    
    def __init__(self, pi=0.05, basis_function=None, multi_bgr=False):
        '''
        '''
        LinearRegressionModel.__init__(self, basis_function)
        if pi < 0.0 or pi > 1:
            raise ValueError('pi must be in the range [0,1].')
        
        self._init_pi = pi
        self._multi_bgr = multi_bgr
        
        
    
    def fit(self, Xp, yp, Xs, ys, itask):
        '''
        '''
        Xp = self._preprocess_data(Xp)
        yp = np.asarray(yp)
        Xs = self._preprocess_data(Xs)
        ys = np.asarray(ys)
        
        if Xp.ndim != 2:
            raise TypeError('Xp must be two dimensional')
        if yp.ndim != 1:
            raise TypeError('yp must be one dimensional')
        if Xs.ndim != 2:
            raise TypeError('Xs must be two dimensional')
        if ys.ndim != 1:
            raise TypeError('ys must be one dimensional')
        if Xp.shape[1] != Xs.shape[1]:
            raise TypeError('Xp and Xs must be the same number of features')
        
        [np,m] = Xp.shape
        Xp = np.c_[np.ones(np), Xp]     
        ns = Xs.shape[0]
        Xs = np.c_[np.ones(ns), Xs]
        
        A = np.linalg.pinv(Xp)
        wp = np.dot(A,yp)
        yp_hat = np.dot(Xp, wp)
        resid = yp-yp_hat
        var = np.sum(resid**2.0) / (np-2.0)
        betap = 1.0/var
        
        mtasks = len(itask)
        if self._multi_bgr:
            pi = self._init_pi*np.ones(mtasks)
            ws = np.zeros((mtasks,m+1))
            betas = np.zeros(mtasks)
            
            for i in xrange(mtasks):
                start = itask[i]
                end = itask[i+1]
                A = np.linalg.pinv(Xs[start:end])
                ws[i] = np.dot(A,ys[start:end])
                ys_hat = np.dot(Xs[start:end], ws[i])
                resid = ys[start:end]-ys_hat
                var = np.sum(resid**2.0) / ((end-start)-2.0)
                betas[i] = 1.0/var
                
        else:
            pi = self._init_pi
            A = np.linalg.pinv(Xs)
            ws = np.dot(A,ys)
            ys_hat = np.dot(Xs, ws)
            resid = ys-ys_hat
            var = np.sum(resid**2.0) / (ns-2.0)
            betas = 1.0/var
        
        self._wp = wp
        self._ws = ws
        self._pi = pi
        self._betap = betap
        self._betas = betas
        
        self._is_init = True
        
    
    def predict(self, X, ret_var=False):
        '''
        '''
        self._init_check()
        X = self._preprocess_data(X)
        yhat = np.dot(X, self._wp)
        if ret_var:
            var = np.ones(len(X))*1/self._betap
            return yhat, var
             
        return yhat
    
    @staticmethod
    def wrap(wp, betap, ws, betas, pi):
        params = np.r_[wp, betap, ws.ravel(), betas, pi]
        return params
    
    @staticmethod
    def unwrap(params, nfeatures, ntasks, multi_bgr=False):
        wp = params[0:nfeatures]
        betap = params[nfeatures]
        
        offset = nfeatures+1
        if multi_bgr:
            ws = params[offset:offset+(nfeatures*ntasks)]
            offset = offset+(nfeatures*ntasks)
            betas = params[offset:offset+ntasks]
            offset = offset+ntasks
            pi = params[offset:offset+ntasks]
        else:
            ws = params[offset:offset+nfeatures]
            betas = params[offset+nfeatures]
            pi = params[offset+nfeatures+1]
                
        return (wp,betap,ws,betas,pi)
        
    class LikelihoodFunction(object):
        
        __slots__ = ('_Xp',
                     '_yp',
                     '_Xs',
                     '_ys',
                     '_itasks',
                     
                     '_multi_bgr'
                     )
        
        def __init__(self, Xp, yp, Xs, ys, itasks, multi_bgr=False):
            
            self._Xp = Xp
            self._yp = yp
            self._Xs = Xs
            self._ys = ys
            
            self._itasks = itasks
            self._multi_bgr
        
        def __call__(self, params):
            '''
            @todo: - error handling
            '''
            Xp = self._Xp
            yp = self._yp
            Xs = self._Xs
            ys = self._ys
            itask = self._itask
            
            ntasks = len(itask)
            [np, m] = Xp.shape
            ns = Xs.shape[0]
            
            
            
            #print 'params'
            #print params
            wp, betap, ws, betas, pi = LinearLRTORegression.unwrap(params, m, ntasks, self._multi_bgr)
            likel_p = -betap/2.0 + np.sum((np.dot(Xp, wp) - yp)**2.0) + np/2*np.log(betap) - np/2.0*np.log(2*np.pi)
            likel_s = 0
            for i in xrange(ntasks):
                start = itask[i]
                end = itask[i+1]
                likel_s += np.log((1-pi[i])*mvnpdf(yp, (np.dot(Xp, wp), betap)) + pi[i]*mvnpdf(ys[start:end], (np.dot(Xs[start:end], ws), betas)))
            
            
            likel = likel_p+likel_s
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

    
if __name__ == '__main__':
    pass