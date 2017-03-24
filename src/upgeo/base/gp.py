'''
Created on Jul 15, 2011

@author: marcel
'''

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

from upgeo.base.likel import GaussianLogLikel, SparseGaussianLogLikel,\
    PITCSparseGaussianLogLikel
from upgeo.base.infer import OnePassInference, ExactInference,\
    FITCOnePassInference, FITCExactInference, PITCOnePassInference
from upgeo.base.selector import RandomSubsetSelector, FixedSelector
from upgeo.util.exception import NotFittedError
from upgeo.amtl.util import gendata_fmtlgp_1d
from scipy.linalg.decomp_cholesky import cho_solve

class GPRegression(BaseEstimator, RegressorMixin):
    '''
    @todo: 
           - define priors on hyperparameters
           - make noise term explicit, that means likel_fun should include this term
           - check whether the gradient and the prediction with a specified mean fct works correctly
    '''
    
    __slots__ = ('_kernel',     #covariance kernel
                 '_meanfct',    #mean function
                 '_likel_fun',  #likelihood function of the model
                 '_infer',      #inference algorithm used to optimize hyperparameters
                 '_priors',     #prior distribution over the parameters
                 
                 '_X',          #covariates of training set
                 '_y',          #targets of training set
                 
                 '_K',          #covariance matrix of the training set
                 '_L',          #cholesky decomposition of the covariance matrix
                 '_alpha',      #weight vector for each data point???
                 
                 '_log_likel',  #log likelihood of the trained model
                           
                 '_d',          #dimension of each input vector
                 '_n',          #number of training samples
                 
                 '_is_init')

    def __init__(self, kernel, meanfct=None, likel_fun=GaussianLogLikel, 
                 infer_method=OnePassInference, priors=None):
        '''
        Constructor
        '''
        if priors != None and len(priors) != kernel.n_params:
            raise ValueError('''number of priors must be equal 
                             to the number of hyperparameters.''')
            
        self._kernel = kernel
        self._meanfct = meanfct
        print likel_fun
        print type(self)
        self._likel_fun = likel_fun(self)
        self._infer = infer_method()
        self._priors = priors    
        
        self._is_init = False
        
    def fit(self, X, y):
        '''
        @todo: parameter check
        '''
        
        self._X = X
        self._y = y
        
        n,d = X.shape
        self._n = n
        self._d = d
         
        infer_method = self._infer
        infer_method.apply(self)
        
        self._is_init = True
        
        return self._log_likel
        
    def predict(self, X, ret_var=False):
        '''
        R contains distances to training and test set 
        @todo: - parameter check
               - check the indexing of R
        '''
        self._init_check()
        
        alpha = self._alpha
        kernel = self._kernel
        meanfct = self._meanfct
        X_train = self._X
        
        ms = np.zeros(len(X))
        if meanfct != None:
            ms = meanfct(X)
        
        Ks = kernel(X, X_train)
        yfit = ms+np.dot(Ks, alpha)
        
        if ret_var:
            L = self._L            
        
            kss = kernel(X, diag=True)
            V = np.linalg.solve(L, Ks.T)
            var = kss - np.sum(V*V,0)
            return (yfit, var)
        
        return yfit
    
    def posterior(self, X):
        '''
        check if this posterior p(f|X) or predictive p(y|X)
        '''
        self._init_check()
        
        L = self._L
        alpha = self._alpha
        kernel = self._kernel
        meanfct = self._meanfct
        X_train = self._X
        
        Ks = kernel(X, X_train)
        Kss = kernel(X)
        
        ms = np.zeros(len(X))
        if meanfct != None:
            ms = meanfct(X)
        
        mean = ms + np.dot(Ks, alpha)
            
        V = np.linalg.solve(L, Ks.T)
        cov = Kss - np.dot(V.T, V)
            
        return (mean, cov)
    
    
    def _get_log_likel(self):
        return self._log_likel
    
    log_likel = property(fget=_get_log_likel)
    
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        params = np.copy(self._kernel.params)
        if self._meanfct != None:
            params = np.r_[params, self._meanfct.params]
        return params
    
    def _set_hyperparams(self, params):
        '''
        '''
        kernel = self._kernel
        meanfct = self._meanfct
        kernel.params = np.copy(params[:kernel.nparams])
        if meanfct != None:
            offset = kernel.nparams
            meanfct = np.copy(params[offset:])
        
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)
    
    def _get_kernel(self):
        return self._kernel
    
    kernel = property(fget=_get_kernel)

    
    def _get_meanfct(self):
        return self._meanfct
    
    meanfct = property(fget=_get_meanfct)
    
    def _get_infer_method(self):
        return self._infer
    
    infer_method = property(fget=_get_infer_method)
    
    def _get_likel_fun(self):
        return self._likel_fun
    
    likel_fun = property(fget=_get_likel_fun)
    
    def _get_training_set(self):
        return self._X, self._y
    
    training_set = property(fget=_get_training_set)
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')

class SparseGPRegression(BaseEstimator, RegressorMixin):
    '''
    @todo: - define priors on hyperparameters
           - make noise term explicit, that means likel_fun should include this term
           - remove the parametrization by likelihood function, because its strongly
             depend on the inference method
           - should we modify the inducing points by hyperparameters update?
           
    @note: wir muessen unterscheiden zwischen signal und noise kernel, damit wir bei der 
           verwendung eines komplexen noise terms unterschieden werden kann, ob es sich
           um die inducing points (noise free) oder datenpunkte handelt. insbesondere
           wuerde eine nicht separierung fehlschlagen unter der verwendung eines gruppenindex
           im datenvektor, der bei der normalen kovarianz berechnung maskiert werden wuerde.
           der gruppenindex spielt nur beim noise term eine rolle. 
           zur zeit ist dies ein hack, da dieses maskierte feature nicht optimiert werden wuerde
           wenn es im signal kernel maskiert ist. Intuitiver waere es das noise model in der likelihood
           function zu deklarieren.
    '''
    
    __slots__ = ('_kernel',         #covariance kernel
                 '_noise_kernel'    #
                 '_meanfct',
                 '_likel_fun',  #likelihood function of the model
                 '_infer',      #inference algorithm used to optimize hyperparameters
                 '_selector',   #subset selection method
                 '_priors',     #prior distribution over the parameters
       
                 '_fix_inducing'#flag if inducing dp are optimized         
                 '_Xu',         #set of the induced datapoints
                 
                 '_X',          #covariates of training set
                 '_y',          #targets of training set
                
                 '_Km',         #covariance matrix of inducing points
                 '_iKm',        #inverse of cov matrix Km
                 '_Lm',         #cholesky decomposition of cov matrix Km
                 
                 '_Kn',         #covariance matrix of training points
                 '_Knm',        #covariance matrix of training and inducing points
                 '_Lnm',
                 
                 '_G',          #G = diag(K-Q)+noise
                 
                 '_V',
                 '_Q',          #symmetric matrix Q of the woodbury identity
                 '_Lq',         #cholesky decomposition of Q
                 '_iQ',         #inverse of Q
                  
                 '_alpha',      #weight vector for each data point???
                 '_B',          #B = iKm - iQ #for covariance prediction
                 
                 '_log_likel',  #log likelihood of the trained model
                           
                 '_d',          #dimension of each input vector
                 '_n',          #number of training samples
                 
                 '_is_init')

    def __init__(self, kernel, noise_kernel=None, meanfct=None, 
                 likel_fun=SparseGaussianLogLikel, 
                 infer_method=FITCOnePassInference, 
                 selector=RandomSubsetSelector(10),
                 priors=None, fix_inducing=True):
        '''
        Constructor
        '''
        if priors != None and len(priors) != kernel.n_params:
            raise ValueError('''number of priors must be equal 
                             to the number of hyperparameters.''')
            
        self._kernel = kernel
        self._noise_kernel = noise_kernel
        self._meanfct = meanfct
        self._likel_fun = likel_fun(self)
        self._infer = infer_method()
        self._selector = selector
        self._priors = priors    
        
        self._fix_inducing = fix_inducing
        self._is_init = False
        
    def fit(self, X, y):
        '''
        @todo: parameter check
        '''
        
        selector = self._selector
        
        #print 'gp_hyperparams={0}'.format(self.hyperparams)
        
        self._Xu = selector.apply(X,y)
        self._X = X
        self._y = y
        
        n,d = X.shape
        self._n = n
        self._d = d
         
        infer_method = self._infer
        infer_method.apply(self)
        
        self._is_init = True
        
        return self._log_likel
        
    def predict(self, X, ret_var=False):
        '''
        R contains distances to training and test set 
        @todo: - parameter check
               - check the indexing of R
        '''
        self._init_check()
        
        print 'gp_pred_hyperparams={0}'.format(self.hyperparams)
        
        alpha = self._alpha
        kernel = self._kernel
        noise_kernel = self._noise_kernel
        meanfct = self._meanfct
        Xu = self._Xu
        
        ms = np.zeros(len(X))
        if meanfct != None:
            ms = meanfct(X)

        
        Ks = kernel(X, Xu)
        yfit = ms + np.dot(Ks, alpha)
        
        if ret_var:
            B = self._B
            kss = kernel(X, diag=True)
            if noise_kernel != None:
                kss = kss + noise_kernel(X, diag=True)
            V = np.sum(Ks*np.dot(B,Ks.T).T,1) #V=diag(Ks*B*Ks')
            #V = np.dot(np.dot(Ks, B), Ks.T)
            var = kss - V
            return (yfit, var)
        
        return yfit
    
    def posterior(self, X, R=None):
        self._init_check()
        
        B = self._B
        alpha = self._alpha
        kernel = self._kernel
        noise_kernel = self._noise_kernel
        meanfct = self._meanfct
        Xu = self._Xu
        
        Ks = kernel(X, Xu)
        Kss = kernel(X)
        if noise_kernel != None:
            Kss = Kss + noise_kernel(X)
        
        ms = np.zeros(len(X))
        if meanfct != None:
            ms = meanfct(X)
        
        mean = ms + np.dot(Ks, alpha)
    
        V = np.dot(np.dot(Ks, B), Ks.T)
        cov = Kss - V
            
        return (mean, cov)
    
    
    def _get_log_likel(self):
        return self._log_likel
    
    log_likel = property(fget=_get_log_likel)
    
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        params = np.copy(self._kernel.params)
        if self._noise_kernel != None:
            params = np.r_[params, self._noise_kernel.params]
        if self._meanfct != None:
            params = np.r_[params, self._meanfct.params]
        return params
    
    def _set_hyperparams(self, params):
        '''
        '''
        kernel = self._kernel
        noise_kernel = self._noise_kernel
        meanfct = self._meanfct
        
        kernel.params = np.copy(params[:kernel.nparams])
        
        offset = kernel.nparams
        if noise_kernel != None:
            noise_kernel.params = np.copy(params[offset:(offset+noise_kernel.nparams)])
            offset += noise_kernel.nparams

        if meanfct != None:
            meanfct = np.copy(params[offset:])
    
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)
    
    def _get_kernel(self):
        return self._kernel
    
    kernel = property(fget=_get_kernel)

    def _get_noise_kernel(self):
        return self._noise_kernel
    
    noise_kernel = property(fget=_get_noise_kernel)
    
    def _get_meanfct(self):
        return self._meanfct
    
    meanfct = property(fget=_get_meanfct)
    
    def _get_infer_method(self):
        return self._infer
    
    infer_method = property(fget=_get_infer_method)
    
    def _get_likel_fun(self):
        return self._likel_fun
    
    likel_fun = property(fget=_get_likel_fun)
    
    def _get_training_set(self):
        return self._X, self._y
    
    training_set = property(fget=_get_training_set)
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')

class PITCSparseGPRegression(BaseEstimator, RegressorMixin):
    '''
    @todo: - define priors on hyperparameters
           - make noise term explicit, that means likel_fun should include this term
           - remove the parametrization by likelihood function, because its strongly
             depend on the inference method
           - should we modify the inducing points by hyperparameters update?
           
    @note: wir muessen unterscheiden zwischen signal und noise kernel, damit wir bei der 
           verwendung eines komplexen noise terms unterschieden werden kann, ob es sich
           um die inducing points (noise free) oder datenpunkte handelt. insbesondere
           wuerde eine nicht separierung fehlschlagen unter der verwendung eines gruppenindex
           im datenvektor, der bei der normalen kovarianz berechnung maskiert werden wuerde.
           der gruppenindex spielt nur beim noise term eine rolle. 
           zur zeit ist dies ein hack, da dieses maskierte feature nicht optimiert werden wuerde
           wenn es im signal kernel maskiert ist. Intuitiver waere es das noise model in der likelihood
           function zu deklarieren.
    '''
    
    __slots__ = ('_kernel',         #covariance kernel
                 '_noise_kernel'    #
                 '_meanfct',
                 '_likel_fun',  #likelihood function of the model
                 '_infer',      #inference algorithm used to optimize hyperparameters
                 '_igroup'
                 '_selector',   #subset selection method
                 '_priors',     #prior distribution over the parameters
       
                 '_fix_inducing'#flag if inducing dp are optimized         
                 '_Xu',         #set of the induced datapoints
                 
                 '_X',          #covariates of training set
                 '_y',          #targets of training set
                
                 '_Km',         #covariance matrix of inducing points
                 '_iKm',        #inverse of cov matrix Km
                 '_Lm',         #cholesky decomposition of cov matrix Km
                 
                 '_Kn',         #covariance matrix of training points
                 '_Knm',        #covariance matrix of training and inducing points
                 '_Lnm',
                 
                 '_G',          #G = diag(K-Q)+noise
                 '_Lg',
                 '_iG',
                 '_iGy',        #iGy = iG*y
                 
                 
                 '_V',
                 '_Q',          #symmetric matrix Q of the woodbury identity
                 '_Lq',         #cholesky decomposition of Q
                 '_iQ',         #inverse of Q
                 
                 '_r',           #r=Kmn*iGy
                 '_alpha',      #weight vector for each data point???
                 '_B',          #B = iKm - iQ #for covariance prediction
                 
                 '_log_likel',  #log likelihood of the trained model
                           
                 '_d',          #dimension of each input vector
                 '_n',          #number of training samples
                 
                 '_is_init')

    def __init__(self, igroup, kernel, noise_kernel=None, meanfct=None, 
                 likel_fun=PITCSparseGaussianLogLikel, 
                 infer_method=PITCOnePassInference,
                 selector=RandomSubsetSelector(10),
                 priors=None, fix_inducing=True):
        '''
        Constructor
        igroup is dependent on the dataset (X,y). pass it to the fit method
        '''
        if priors != None and len(priors) != kernel.n_params:
            raise ValueError('''number of priors must be equal 
                             to the number of hyperparameters.''')
            
        self._kernel = kernel
        self._noise_kernel = noise_kernel
        self._meanfct = meanfct
        self._likel_fun = likel_fun(self)
        self._infer = infer_method()
        self._selector = selector
        self._priors = priors  
        self._igroup = igroup  
        
        self._fix_inducing = fix_inducing
        self._is_init = False
        
    def fit(self, X, y):
        '''
        @todo: parameter check
        '''
        
        selector = self._selector
        
        #print 'gp_hyperparams={0}'.format(self.hyperparams)
        
        self._Xu = selector.apply(X,y)
        self._X = X
        self._y = y
        
        n,d = X.shape
        self._n = n
        self._d = d
         
        infer_method = self._infer
        infer_method.apply(self)
        
        self._is_init = True
        
        return self._log_likel
        
    def predict(self, X, ret_var=False, blocks=None):
        '''
        R contains distances to training and test set 
        @todo: - parameter check
               - check the indexing of R
               - check pic approximation
        '''
        self._init_check()
        
        print 'gp_pred_hyperparams={0}'.format(self.hyperparams)
        
        alpha = self._alpha
        kernel = self._kernel
        noise_kernel = self._noise_kernel
        meanfct = self._meanfct
        Xu = self._Xu
        
        ms = np.zeros(len(X))
        if meanfct != None:
            ms = meanfct(X)

        
        if blocks == None:
            #PITC predictive mean
            Ks = kernel(X, Xu)
            yfit = ms + np.dot(Ks, alpha)
                
        
            if ret_var:
                B = self._B
                kss = kernel(X, diag=True)
                if noise_kernel != None:
                    kss = kss + noise_kernel(X, diag=True)
                V = np.sum(Ks*np.dot(B,Ks.T).T,1) #V=diag(Ks*B*Ks')
                #V = np.dot(np.dot(Ks, B), Ks.T)
                var = kss - V
                return (yfit, var)

            return yfit
        else:
            #PIC predictive mean
            Xt = self._X
            n = len(Xt)
            m = len(Xu)
            igroup = self._igroup
            k = len(igroup)
            igroup = np.r_[igroup, n]
            
            Kn = self._Kn
            Km = self._Km
            iKm = self._iKm
            G = self._G
            iG = self._iG
            iGy = self._iGy
            
            
            yfit = np.zeros(len(X)) 
            yfit1 = np.zeros(len(X))
            var = np.zeros(len(X))
            
            Qn = np.dot(np.dot(self._Knm, iKm), self._Knm.T)
            #Kn = np.zeros((n,n)) 
            for i in xrange(k):
                start = igroup[i]
                end = igroup[i+1]
                Qn[start:end,start:end] = self.kernel(X[start:end],X[start:end])
                if self._noise_kernel != None:
                    Qn[start:end,start:end] += self.noise_kernel(X[start:end],X[start:end])
                    
            iQn = np.linalg.inv(Qn)
            iQny = np.dot(iQn, self._y)
            
            
            for i in xrange(k):
                Knm = self._Knm
                
                start = igroup[i]
                end = igroup[i+1]
                iblock = blocks == i
                
                if sum(iblock) > 0:                    
                    #Knm = self._Knm[np.r_[0:start,end:n]]
                    y = self._y[np.r_[0:start,end:n]]
                    yb = self._y[start:end]
                    
                    Ksb = kernel(X[iblock], Xt[start:end])
                    Ks = kernel(X[iblock], Xu)
                    
                    Kss = np.zeros((len(X), n))
                    Kss[iblock,start:end] = Ksb
                    print np.dot(np.dot(Ks, iKm), Knm[np.r_[0:start,end:n]].T).shape
                     
                    Kss[:,np.r_[0:start,end:n]] = np.dot(np.dot(Ks, iKm), Knm[np.r_[0:start,end:n]].T)
                    print Kss
                    
                    B1 = np.zeros((Knm.shape[0],m))
                    U = np.zeros((m,m))
                    r = np.zeros(m)
                    for j in xrange(k):
                        if i != j:
                            startj = igroup[j]
                            endj = igroup[j+1]
                            #B1[startj:endj] = np.dot(Knm[startj:endj].T, iG[j]).T
                            U += np.dot(np.dot(Knm[startj:endj].T, iG[j]), Knm[startj:endj])
                            r += np.dot(Knm[startj:endj].T, iGy[startj:endj])
                            
                    #B1 = B1[np.r_[0:start,end:n]] 
                    #Knm = Knm[np.r_[0:start,end:n]]
                            
                    
                    Q = Km + U#np.dot(B1.T, Knm)
                    Lq = np.linalg.cholesky(Q)
                    iQ = cho_solve((Lq, 1), np.eye(m))
                    alpha = np.dot(iQ, r)
                    
                    Ln = np.linalg.cholesky(Kn[i])
                    alpha_b = cho_solve((Ln, 1), yb)
                    #alpha_b = np.dot(np.linalg.inv(self._Kn[i]), yb)
                        
                    yfit[iblock] = np.dot(Ks,alpha)+np.dot(Ksb, alpha_b)    
                    yfit1[iblock] = np.dot(Kss, iQny)
                    print 'yfit={0}'.format(yfit)
                    print 'yfit1={0}'.format(yfit1)
                    print 'iQny={0}'.format(iQny)
                    print 'alphab={0}'.format(alpha_b)
                    if ret_var:
                        #B = self._B
                        kss  = kernel(X[iblock], diag=True)
                        if noise_kernel != None:
                            kss = kss + noise_kernel(X[iblock], diag=True)
        
                        B = iKm - iQ
                        W = np.linalg.solve(Ln, Ksb.T)
                        V = np.sum(Ks*np.dot(B,Ks.T).T,1) + np.sum(W*W,0) #+ np.sum(Ksb*np.dot(iG[i], Ksb.T).T,1) #V=diag(Ks*B*Ks')
                        
                        #V = np.dot(np.dot(Ks, B), Ks.T)
                        var[iblock] = kss - V
                        #return (yfit, var)                
        
        return yfit if ret_var == False else (yfit, var) 
    
    def posterior(self, X, R=None):
        '''
        compute the posterior for pic approximation
        '''
        self._init_check()
        
        B = self._B
        alpha = self._alpha
        kernel = self._kernel
        noise_kernel = self._noise_kernel
        meanfct = self._meanfct
        Xu = self._Xu
        
        Ks = kernel(X, Xu)
        Kss = kernel(X)
        if noise_kernel != None:
            Kss = Kss + noise_kernel(X)
        
        ms = np.zeros(len(X))
        if meanfct != None:
            ms = meanfct(X)
        
        mean = ms + np.dot(Ks, alpha)
    
        V = np.dot(np.dot(Ks, B), Ks.T)
        cov = Kss - V
            
        return (mean, cov)
    
    
    def _get_log_likel(self):
        return self._log_likel
    
    log_likel = property(fget=_get_log_likel)
    
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        params = np.copy(self._kernel.params)
        if self._noise_kernel != None:
            params = np.r_[params, self._noise_kernel.params]
        if self._meanfct != None:
            params = np.r_[params, self._meanfct.params]
        return params
    
    def _set_hyperparams(self, params):
        '''
        '''
        kernel = self._kernel
        noise_kernel = self._noise_kernel
        meanfct = self._meanfct
        
        kernel.params = np.copy(params[:kernel.nparams])
        
        offset = kernel.nparams
        if noise_kernel != None:
            noise_kernel.params = np.copy(params[offset:(offset+noise_kernel.nparams)])
            offset += noise_kernel.nparams

        if meanfct != None:
            meanfct = np.copy(params[offset:])
    
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)
    
    def _get_kernel(self):
        return self._kernel
    
    kernel = property(fget=_get_kernel)

    def _get_noise_kernel(self):
        return self._noise_kernel
    
    noise_kernel = property(fget=_get_noise_kernel)
    
    def _get_meanfct(self):
        return self._meanfct
    
    meanfct = property(fget=_get_meanfct)
    
    def _get_infer_method(self):
        return self._infer
    
    infer_method = property(fget=_get_infer_method)
    
    def _get_likel_fun(self):
        return self._likel_fun
    
    likel_fun = property(fget=_get_likel_fun)
    
    def _get_training_set(self):
        return self._X, self._y
    
    training_set = property(fget=_get_training_set)
    
    def _init_check(self):
        '''
        '''
        if not self._is_init:
            raise NotFittedError('fit was not invoked before')
    

if __name__ == '__main__':
    import time
    
    from upgeo.util.metric import mspe
    from upgeo.base.kernel import NoiseKernel, SEKernel, ARDSEKernel
    from upgeo.base.util import plot1d_gp, gendata_1d, f1, f2
    import matplotlib.pyplot as plt    
    #(X,y) = gendata_1d(f2, 0, 3, 100, 1)
    #X = X[:,np.newaxis]
    
    kernel = SEKernel(np.log(1), np.log(1))# + NoiseKernel(np.log(0.5))
    x, yp, ys = gendata_fmtlgp_1d(kernel, kernel, -5, 5, 100, 4, sigma=0.5, seed=38472)
    #print check_kernel_gradient(kernel, X)
    plt.plot(x, yp, '*')
    plt.show()
    kernel = SEKernel(np.log(0.1), np.log(1)) + NoiseKernel(np.log(0.5))
    gp = GPRegression(kernel, None, GaussianLogLikel, ExactInference)
#    t = time.time()
    gp.fit(x, yp)
    #print 'fit_time: {0}'.format(time.time()-t)
    plot1d_gp(gp, -5, 5)
#    #print gp.log_likel
#    t = time.time()
#    gp.likel_fun()
#    print 'likel_time: {0}'.format(time.time()-t)
#    t = time.time()
#    gp.likel_fun.gradient()
#    print 'grad_time: {0}'.format(time.time()-t)
#    print np.exp(gp.hyperparams)
#    print gp.log_likel
#    print gp.likel_fun.gradient()
    
    (X,y) = gendata_1d(f1, -3, 3, 50, 0.01)
    #X = X[:,np.newaxis]
    
    #kernel = SqConstantKernel(np.log(10)) + SEKernel(np.log(1.4), np.log(1.7)) + NoiseKernel(np.log(0.2))
    
    kernel = SEKernel(np.log(0.5), np.log(1)) + NoiseKernel(np.log(0.5))
    t = time.time()
    gp = GPRegression(kernel, None, GaussianLogLikel, ExactInference)
   #gp.fit(X, y)
    #print 'opt_time: {0}'.format(time.time()-t)
    #plot1d_gp(gp, 0, 3)
    
    #kernel = ARDSEKernel([np.log(0.5)], np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = SEKernel(np.log(0.58486524), np.log(0.90818889)) + NoiseKernel(np.log(0.4807206))
    Xu = np.linspace(0,3,20)
    Xu = Xu[:,np.newaxis]
    #selector = RandomSubsetSelector(10)
    selector = FixedSelector(Xu)
    gp = SparseGPRegression(kernel, infer_method=FITCOnePassInference, selector=selector)
    t = time.time()
    gp.fit(X, y)
    print 'fit_time: {0}'.format(time.time()-t)
    #plot1d_gp(gp, 0, 3)
    #print gp.log_likel
    t = time.time()
    gp.likel_fun()
    print 'likel_time: {0}'.format(time.time()-t)
    t = time.time()
    gp.likel_fun.gradient()
    print 'grad_time: {0}'.format(time.time()-t)
    print np.exp(gp.hyperparams)
    print gp.log_likel
    print gp.likel_fun.gradient()
    
    t = time.time()
    gp = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=True)
    gp.fit(X, y)
    print 'opt_time: {0}'.format(time.time()-t)
    plot1d_gp(gp, 0, 3)
