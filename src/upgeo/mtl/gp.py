'''
Created on Sep 24, 2012

@author: marcel
'''
import numpy as np

from sklearn.base import RegressorMixin, BaseEstimator
from upgeo.mtl.likel import CMOGPGaussianLogLikel, SparseCMOGPGaussianLogLikel
from upgeo.mtl.infer import CMOGPOnePassInference, CMOGPExactInference,\
    SparseCMOGPOnePassInference, SparseCMOGPExactInference
from upgeo.util.exception import NotFittedError
from upgeo.base.gp import GPRegression
from upgeo.base.infer import ExactInference, OnePassInference
from upgeo.base.likel import GaussianLogLikel
from upgeo.util.glob import APPROX_TYPE
from upgeo.base.selector import RandomSubsetSelector, FixedSelector
from upgeo.base.kernel import GaussianKernel, CompoundKernel
from scipy.linalg.decomp_cholesky import cho_solve
import scipy
from scipy.stats.stats import cov


class CMOGPRegression(BaseEstimator, RegressorMixin):
    '''
    @todo: - allow to specify a mean function
           - define priors on hyperparameters
           - make noise term explicit, that means likel_fun should include this term
           
    '''
    
    __slots__ = (
                 '_kernel',    #covariance function is AMTLKernel
                 #'_mtp',        #number of hyperparameters of primary kernel        
                 #'_mts',        #number of hyperparameters of the secondary kernels
                 
                 '_d',          #dimension of each input vector
                 '_n',          #total number of training samples
                 '_ntasks',     #number of secondary tasks
                 '_task_sizes', #array of sizes for each task         

                 
                 '_likel_fun',  #likelihood function of the model
                 '_infer',      #inference algorithm used to optimize hyperparameters
                 
                 '_X',          #covariates of training set 
                 '_y',
                 '_itask',      #tasks indices denotes the starting index of each specific task data in Xs, ys
        
                 '_Kf',         #covariance matrix arising from convolution
                 '_Ks',         #covariance matrix associated with the independent process
                 '_K',          #K = Kf+Ks
                 '_L',          #L = chol(K)
                 '_iK'          #iK = inv(K)
                 '_alpha'       #alpha = iK*y
        
                 
                 '_log_likel',  #log likelihood of the trained model
                           
                 
                 '_is_init')

    def __init__(self, kernel, likel_fun=CMOGPGaussianLogLikel,  
                 infer_method=CMOGPOnePassInference):
        '''
        Constructor
        '''
            
        self._kernel = kernel
        #self._mtp = pKernel.n_params
        #self._mts = sKernel.n_params
        self._likel_fun = likel_fun(self)
        self._infer = infer_method()
        
        self._is_init = False


    def fit(self, X, y, itask):
        '''
        @todo: parameter check
        '''
        
        self._X = X
        self._y = y
        
        n,d = X.shape
        ntasks = len(itask)
        
        self._d = d
        self._n = n
        self._ntasks = ntasks
        self._itask = itask
        
        self._task_sizes = np.diff(np.r_[itask, n])
        
        #initialize the task specific kernels and 
        #the correlation vector to the primary task
         
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
        #import time
        #self._init_check()
    
        X = np.atleast_2d(X)
        
        kernel = self._kernel
        Xt = self._X
        itaskXt = self._itask
        ntasks = self._ntasks
        alpha = self._alpha
         
        #preprocess test data
        n = len(X)
        X = np.tile(X, (ntasks,1))
        itaskX = np.arange(0, n*ntasks, n)
        
        Ks = kernel.cov(Xt, X, itaskXt, itaskX)
        
        #print 'Ks'
        #print Ks
        #print X
        #print Xt
        #sio.savemat('/home/marcel/mogpKs.mat', {'Ks': Ks})
        yfit = np.dot(Ks.T, alpha)
        yfit = np.reshape(yfit, (ntasks, n)).T
        
        if ret_var:
            '''
            '''
            L = self._L
            #data processing to compute the diag cov matrix for the test points 
            kss = kernel.cov(X, itaskX=itaskX, diag=True)
            V = np.linalg.solve(L, Ks)
            var = kss - np.sum(V*V,0)
            var = np.reshape(var, (ntasks, n)).T
            return (yfit, var)
        
        return yfit

    def predict_task(self, X, q, ret_var=False):
        '''
        '''
        #import time
        #self._init_check()
        ntasks = self._ntasks
        itask = self._itask
        
        if q > ntasks:
            raise ValueError('Unknown task: {0}'.format(q))
    
        X = np.atleast_2d(X)
        kernel = self._kernel
    
        Xt = self._X
        alpha = self._alpha
        
        Ks = kernel.cov(Xt, X, itask, q=q)
        
        #TODO: remove this computation to the mtl kernel
#        n = self._n
#        m = len(X)
#        itask = np.r_[itask, n]
#        Ks = np.zeros((n,m))
#        for i in xrange(ntasks):
#            start = itask[i]
#            end = itask[i+1]
#            Ks[start:end,:] = kernel.cross_cov(Xt[start:end,:], X, i, q)
#            #hack, should be computed internally
#            if i == q:
#                idpKernel = kernel.independent_kernel(q)
#                if idpKernel != None:
#                    Ks[start:end,:] += idpKernel(Xt[start:end,:], X)
                    
                
        yfit = np.dot(Ks.T, alpha)
        if ret_var:
            '''
            '''
            L = self._L
            kss  = kernel.cov_block(X, i=q, diag=True)
            V = np.linalg.solve(L, Ks)
            var = kss - np.sum(V*V,0)
            return (yfit, var)
        
        return yfit    

    def _get_log_likel(self):
        return self._log_likel
    
    log_likel = property(fget=_get_log_likel)
    
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        params = np.copy(self._kernel.params)
        #if self._meanfct != None:
        #    params = np.r_[params, self._meanfct.params]
        return params
    
    def _set_hyperparams(self, params):
        '''
        '''
        kernel = self._kernel
        #meanfct = self._meanfct
        kernel.params = params
        #if meanfct != None:
        #    offset = kernel.nparams
        #    meanfct = np.copy(params[offset:])
        
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)
    
    def _get_kernel(self):
        return self._kernel
    
    kernel = property(fget=_get_kernel)
    
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

class SparseCMOGPRegression(BaseEstimator, RegressorMixin):
    '''
    @todo: - allow to specify a mean function
           - define priors on hyperparameters
           - make noise term explicit, that means likel_fun should include this term
           
    '''
    
    __slots__ = (
                 '_kernel',    #covariance function is AMTLKernel 
                 '_beta' 
                 #'_mtp',        #number of hyperparameters of primary kernel        
                 #'_mts',        #number of hyperparameters of the secondary kernels
                 
                 
                 '_d',          #dimension of each input vector
                 '_n',          #total number of training samples
                 '_ntasks',     #number of secondary tasks
                 '_task_sizes', #array of sizes for each task         

                 
                 '_likel_fun',  #likelihood function of the model
                 '_infer',      #inference algorithm used to optimize hyperparameters
                 
                 
                 '_selector',
                 '_approx_type',
                 '_fix_inducing',
                 
                 '_X',          #covariates of training set
                 '_Xu', 
                 '_y',
                 '_itask',      #tasks indices denotes the starting index of each specific task data in Xs, ys
        
                 '_Kf',
                 '_Ku',        #
                 '_Lu',
                 '_D',
                 '_Ld',
                 '_A',
                 '_La',
                 '_alpha',      
                 
                 '_log_likel',  #log likelihood of the trained model
                           
                 
                 '_is_init')

    def __init__(self, kernel, beta=None, likel_fun=SparseCMOGPGaussianLogLikel,  
                 infer_method=SparseCMOGPOnePassInference, 
                 approx_type=APPROX_TYPE.PITC, selector=RandomSubsetSelector(10),
                 fix_inducing=True):
        '''
        Constructor
        '''
            
        self._kernel = kernel
        self._beta = beta
        self._likel_fun = likel_fun(self)
        self._infer = infer_method()
        self._approx_type = approx_type
        self._selector = selector
        self._fix_inducing = fix_inducing
        self._is_init = False


    def fit(self, X, y, itask):
        '''
        @todo: parameter check
        '''
        
        selector = self._selector
        #Xu = np.arange(-5,5)
        self._Xu = selector.apply(X,y)
        #Xu = Xu[:,np.newaxis]
        #self._Xu = Xu
        self._X = X
        self._y = y
        
        n,d = X.shape
        ntasks = len(itask)
        
        self._d = d
        self._n = n
        self._ntasks = ntasks
        self._itask = itask
        
        self._task_sizes = np.diff(np.r_[itask, n])
        
        if self._beta != None:
            self._beta = np.repeat(self._beta, ntasks)
        
        
        #initialize the task specific kernels and 
        #the correlation vector to the primary task
         
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
        #import time
        #self._init_check()
    
        X = np.atleast_2d(X)
        
        kernel = self._kernel
        beta = self._beta
        Xu = self._Xu
        ntasks = self._ntasks
       
        #kernel.params = np.log(np.array([ 0.73195154, 0.00469354, 0.30550024, 0.08275141, 0.49153723, 1.09938963, 0.26688294, 0.54233881, 0.53488461, 0.32496261]))
        #kernel.params = np.log(np.array([ 1.73195154, 1, 0.0150024, 0.01275141, 0.49153723, 0.39938963, 1.26688294, 0.54233881, 1.0488461, 0.32496261]))
        #self._infer.apply(self)
        #print 'alpha'
        #print alpha
        
        alpha = self._alpha
        
        #preprocess test data
        n = len(X)
        X = np.tile(X, (ntasks,1))
        itask = np.arange(0, n*ntasks, n)
        
        #similiar to the predictive distribution of the alvarez paper
        Ksu = kernel.cov(X, Xu, itask)
        yfit = np.dot(Ksu, alpha)
        
        
        Kfu = self._Kfu
        iD = self._iD
        iDy = self._iDy
        Xt = self._X
        itaskXt = self._itask
        k = self._ntasks
        
        
        #Kfs = kernel.cov(Xt, X, itaskXt, itask)
        Kfs = np.zeros((len(Xt), len(X)))
        U = np.zeros((len(Xu), len(Xt)))
        for i in xrange(k):
            startXt = itaskXt[i]
            endXt = itaskXt[i+1] if i != k-1 else len(Xt)
            start = itask[i]
            end = itask[i+1] if i != k-1 else len(X)
            idpKernel = kernel.independent_kernel(i)
            Kfs[startXt:endXt,start:end] = idpKernel(Xt[startXt:endXt], X[start:end]) 
            U[:,startXt:endXt] = np.dot(Kfu[startXt:endXt].T, iD[i])
        yfit += np.dot(Kfs.T, iDy-np.dot(U.T, alpha))
        
        yfit = np.reshape(yfit, (ntasks, n)).T
        
        #W = np.zeros((len(self._X),len(self._X)))
        
        #Q = np.zeros((len(X),len(X)))
#        for i in xrange(ntasks):
#            start = self._itask[i]
#            end = self._itask[i+1] if i != ntasks-1 else len(self._X)
#            
#            print 'Di={0}'.format(self._D[i].shape)
#            print W[start:end,start:end].shape    
#            W[start:end,start:end] = self._D[i]
#            
        #Q = W + np.dot(np.dot(self._Kfu, self._iKu), self._Kfu.T)
        
        
        if ret_var:
            '''
            #@todo: structure of Ds depends on the choosen sparse approximation type. 
            #but it also possible to compute the full matrix. make it possible per flag.    
            '''
            approx = self._approx_type
            iA = self._iA
            iKu = self._iKu
            B = iKu - iA
            Ks = kernel.cov(X, itaskX=itask, diag=True)
            #approx = self._approx_type
            #print np.dot(B,Ks.T)
            V = np.sum(Ksu*np.dot(B,Ksu.T).T,1) #V=diag(Ks*B*Ks')
            var = Ks-V
            
            #check if is correct if we use for UiAU the the diagonal structure of the
            #approximation or the full matrix
            if approx == APPROX_TYPE.FITC:
                UiAU = np.sum((U*iA)*U,1)
                V1 = np.sum((Kfs*(iD-UiAU))*Kfs,1)
            elif approx == APPROX_TYPE.PITC:
                m = ntasks*n
                V1 = np.zeros((m))
                for i in xrange(k):
                    start = itaskXt[i]
                    end = itaskXt[i+1] if i != k-1 else len(Xt) 
                    UiAU = np.dot(np.dot(U[:,start:end].T, iA), U[:,start:end])
                    
                    V1 += np.sum(np.dot(Kfs[start:end].T, iD[i]-UiAU)*Kfs[start:end].T,1)
            UiA = np.dot(U.T,iA)
            V2 = np.sum(Kfs*np.dot(UiA, Ksu.T),0)
             
            var -= V1 + 2.0*V2
            var = np.reshape(var, (ntasks, n)).T
            if beta != None:
                var = var + 1.0/beta
            return (yfit, var)
        
        return yfit

    def predict_task(self, X, q, ret_var=False):
        '''
        '''
        #import time
        #self._init_check()
        ntasks = self._ntasks
        itask = self._itask
        
        if q > ntasks:
            raise ValueError('Unknown task: {0}'.format(q))
    
        X = np.atleast_2d(X)
        n = len(X)
        kernel = self._kernel
        beta = self._beta
    
        Xu = self._Xu
        alpha = self._alpha
                
        Ksu = kernel.cov_block(X, Xu, q)
        #Ksu = kernel.cov(X, Xu, itask)
        yfit = np.dot(Ksu, alpha)
        
        Kfu = self._Kfu
        iD = self._iD
        iDy = self._iDy
        Xt = self._X
        itaskXt = self._itask
        k = self._ntasks
        
        Ksu = kernel.cov_block(X, Xu, q)
        #print 'Ksu={0}'.format(Ksu)
        #Ksu1 = kernel.cov_block(Xu, X, q)
        #print 'Ksu1={0}'.format(Ksu-Ksu1.T)

        Kfs = np.zeros((len(Xt), len(X)))
        U = np.zeros((len(Xu), len(Xt)))
        for i in xrange(k):
            start = itaskXt[i]
            end = itaskXt[i+1] if i != k-1 else len(Xt)
            if i == q:
                idpKernel = kernel.independent_kernel(i)
                Kfs[start:end] = idpKernel(Xt[start:end], X)
                      
            U[:,start:end] = np.dot(Kfu[start:end].T, iD[i])
        yfit += np.dot(Kfs.T, iDy-np.dot(U.T, alpha))
          
    
        if ret_var:
            '''
            '''

            approx = self._approx_type
            iA = self._iA
            iKu = self._iKu
            B = iKu - iA
            Ks = kernel.cov_block(X, i=q, diag=True)
            #approx = self._approx_type
            UL = Ksu*np.dot(B,Ksu.T).T
            axis = 1 if n > 1 else 0
            V = np.sum(Ksu*np.dot(B,Ksu.T).T,axis) #V=diag(Ks*B*Ks')
            var = Ks-V
            
            #check if is correct if we use for UiAU the the diagonal structure of the
            #approximation or the full matrix
            if approx == APPROX_TYPE.FITC:
                UiAU = np.sum((U*iA)*U,1)
                V1 = np.sum((Kfs*(iD-UiAU))*Kfs,1)
            elif approx == APPROX_TYPE.PITC:
                V1 = np.zeros(n)
                for i in xrange(k):
                    start = itaskXt[i]
                    end = itaskXt[i+1] if i != k-1 else len(Xt) 
                    UiAU = np.dot(np.dot(U[:,start:end].T, iA), U[:,start:end])
                    
                    V1 += np.sum(np.dot(Kfs[start:end].T, iD[i]-UiAU)*Kfs[start:end].T,1)
            else:
                raise TypeError('Unknown approx method')
            
            UiA = np.dot(U.T,iA)
            
            #axis = 0 if n > 1 else 1
            V2 = np.sum(np.squeeze(Kfs)*np.dot(UiA, Ksu.T),0)
             
            var -= V1 + 2.0*V2
            if beta != None:
                var += 1.0/beta[q]
            return (yfit, var)
            
        return yfit    

    def posterior(self, X):
        '''
        R contains distances to training and test set 
        @todo: - parameter check
               - check the indexing of R
        '''
        #import time
        #self._init_check()
    
        X = np.atleast_2d(X)
        
        kernel = self._kernel
        beta = self._beta
        Xu = self._Xu
        ntasks = self._ntasks
        
        #kernel.params = np.log(np.array([ 0.73195154, 0.00469354, 0.30550024, 0.08275141, 0.49153723, 1.09938963, 0.26688294, 0.54233881, 0.53488461, 0.32496261]))
        #kernel.params = np.log(np.array([ 1.73195154, 1, 0.0150024, 0.01275141, 0.49153723, 0.39938963, 1.26688294, 0.54233881, 1.0488461, 0.32496261]))
        #self._infer.apply(self)
        #print 'alpha'
        #print alpha
        
        alpha = self._alpha
        
        #preprocess test data
        n = len(X)
        print 'X={0}'.format(X)
        X = np.tile(X, (ntasks,1))
        itask = np.arange(0, n*ntasks, n)
        
        
        #similiar to the predictive distribution of the alvarez paper
        Ksu = kernel.cov(X, Xu, itask)
        mu = np.dot(Ksu, alpha)

        
        
        Kfu = self._Kfu
        iD = self._iD
        iDy = self._iDy
        Xt = self._X
        itaskXt = self._itask
        k = self._ntasks
        
        Kfs = np.zeros((len(Xt), len(X)))
        U = np.zeros((len(Xu), len(Xt)))
        for i in xrange(k):
            startXt = itaskXt[i]
            endXt = itaskXt[i+1] if i != k-1 else len(Xt)
            start = itask[i]
            end = itask[i+1] if i != k-1 else len(X)
            idpKernel = kernel.independent_kernel(i)
            
            Kfs[startXt:endXt,start:end] = idpKernel(Xt[startXt:endXt], X[start:end]) 
            U[:,startXt:endXt] = np.dot(Kfu[startXt:endXt].T, iD[i])
        mu += np.dot(Kfs.T, iDy-np.dot(U.T, alpha))
  
        #mu = np.reshape(mu, (ntasks, n)).T
        
        #W = np.zeros((len(self._X),len(self._X)))
        
        #Q = np.zeros((len(X),len(X)))
#        for i in xrange(ntasks):
#            start = self._itask[i]
#            end = self._itask[i+1] if i != ntasks-1 else len(self._X)
#            
#            print 'Di={0}'.format(self._D[i].shape)
#            print W[start:end,start:end].shape    
#            W[start:end,start:end] = self._D[i]
#            
        #Q = W + np.dot(np.dot(self._Kfu, self._iKu), self._Kfu.T)
        
        
        '''
        #@todo: structure of Ds depends on the choosen sparse approximation type. 
        #but it also possible to compute the full matrix. make it possible per flag.    
        '''
        iA = self._iA
        iKu = self._iKu
        B = iKu - iA
        Ks = kernel.cov(X, itaskX=itask)
        #approx = self._approx_type
        #print np.dot(B,Ks.T)
        #V = np.sum(Ksu*np.dot(B,Ksu.T).T,1) #V=diag(Ks*B*Ks')
        V = np.dot(np.dot(Ksu, B), Ksu.T)
        Sigma = Ks-V
        
        m = ntasks*n
        V1 = np.zeros((m,m))
        for i in xrange(k):
            start = itaskXt[i]
            end = itaskXt[i+1] if i != k-1 else len(Xt) 
            UiAU = np.dot(np.dot(U[:,start:end].T, iA), U[:,start:end])
            V1 += np.dot(np.dot(Kfs[start:end].T, iD[i]-UiAU), Kfs[start:end])
            
        
        UiA = np.dot(U.T,iA)
        V2 = np.dot(np.dot(Kfs.T, UiA),Ksu.T)

        Sigma -= V1+V2+V2.T
        if beta != None: 
            Sigma = Sigma + np.diag(1.0/np.repeat(beta, n))
        
        #Sigma = np.reshape(var, (n,2*ntasks,n)).T
        #var = np.reshape(Sigma, (ntasks, n)).T
        return (mu, Sigma)
        
        return yfit

    def _get_log_likel(self):
        return self._log_likel
    
    log_likel = property(fget=_get_log_likel)
    
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        params = np.copy(self._kernel.params)
        #if self._meanfct != None:
        #    params = np.r_[params, self._meanfct.params]
        return params
    
    def _set_hyperparams(self, params):
        '''
        '''
        kernel = self._kernel
        #meanfct = self._meanfct
        kernel.params = params
        #if meanfct != None:
        #    offset = kernel.nparams
        #    meanfct = np.copy(params[offset:])
        
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)
    
    def _get_kernel(self):
        return self._kernel
    
    kernel = property(fget=_get_kernel)
    
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

    
class STLGPRegression(BaseEstimator, RegressorMixin):
    '''
    Wrapper class by using basic GP in a MTL setting by using only training data of the
    primary task
    '''
    __slots__ = ('_likel_fun',
                 '_infer_method',
                 '_kernel',
                 '_priors',
                 '_gpmodels',
                 '_ntasks')
    
    def __init__(self, kernel, likel_fun=GaussianLogLikel, 
                 infer_method=OnePassInference, priors=None):
        
        self._likel_fun = likel_fun
        self._infer_method = infer_method
        self._kernel = kernel
        self._priors = priors
    
    def fit(self, X, y, itask):
        '''
        '''
        
        n = X.shape[0]
        ntasks = len(itask)
        itask = np.r_[itask, n]
        gpmodels = np.empty(ntasks, dtype=np.object)
        for i in xrange(ntasks):
            start = itask[i]
            end = itask[i+1]
            
            gp = GPRegression(self._kernel.copy(), likel_fun=self._likel_fun, 
                              infer_method=self._infer_method, 
                              priors=self._priors)
            print 'Xshape'
            print X[start:end].shape
            gp.fit(X[start:end], y[start:end])
            gpmodels[i] = gp
            
        self._ntasks = ntasks
        self._gpmodels = gpmodels
    
    def predict(self, X, ret_var=False):
        '''
        '''
        gpmodels = self._gpmodels
        ntasks = self._ntasks
        n = X.shape[0]
        
        yfit = np.empty((n,ntasks))
        if ret_var:
            var =  np.empty((n,ntasks))
        for i in xrange(ntasks):
            gp = gpmodels[i]
            if ret_var:
                yfit[:,i], var[:,i] = gp.predict(X, True)
            else:
                yfit[:,i] = gp.predict(X, False)
        
        return (yfit, var) if ret_var else yfit


    def predict_task(self, X, q, ret_var=False):
        gp = self._gpmodels[q]
        return gp.predict(X, ret_var)
       
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        return self._gp.hyperparams
    
    def _set_hyperparams(self, params):
        '''
        '''
        self._gp.hyperparams = params
        
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)
    
    def _get_log_likel(self):
        ntasks = self._ntasks
        gpmodels = self._gpmodels
        ll = 0
        for i in xrange(ntasks):
            gp = gpmodels[i]
            ll += gp.log_likel
            
        return ll
    
    
    log_likel = property(fget=_get_log_likel)

class PooledGPRegression(BaseEstimator, RegressorMixin):
    '''
    Wrapper class by using basic GP in a MTL setting by using only training data of the
    primary task
    '''
    __slots__ = ('_gp',
                 '_ntasks')
    
    def __init__(self, kernel, likel_fun=GaussianLogLikel, 
                 infer_method=OnePassInference, priors=None):
    
        self._gp = GPRegression(kernel, likel_fun=likel_fun, infer_method=infer_method, priors=priors)
    
    def fit(self, X, y, itask):
        '''
        '''
        self._gp.fit(X, y)
        ntasks = len(itask)
        self._ntasks = ntasks
    
    def predict(self, X, ret_var=False):
        '''
        '''
        gp = self._gp
        ntasks = self._ntasks
        
        if ret_var:
            yfit, var = gp.predict(X, True)
            yfit = np.tile(yfit, (1,ntasks))
            var = np.tile(var, (1,ntasks))
        else:
            yfit = gp.predict(X, False)
            yfit = np.tile(yfit, (1,ntasks))
        
        return (yfit, var) if ret_var else yfit


    def predict_task(self, X, q, ret_var=False):
        gp = self._gp
        return gp.predict(X, ret_var)
       
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        return self._gp.hyperparams
    
    def _set_hyperparams(self, params):
        '''
        '''
        self._gp.hyperparams = params
        
    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)

    def _get_log_likel(self):
        return self._gp.log_likel
    
    log_likel = property(fget=_get_log_likel)
    
#        #t = time.time()
#        kernel = self._kernel
#        #latent kernel + priate kernel
#        Kpstar = kernel.cross_cov(X, Xp, task=0, latent=True) + kernel.cross_cov(X, Xp, task=0, latent=False)
#        Ksstar = np.zeros((n, ms))
#        for i in xrange(mtasks):
#            start = itask[i]
#            end = itask[i+1]
#            Ksstar[:,start:end] = kernel.cross_cov(X, Xs[start:end], task=i+1, latent=True)
#            
#        yfit = np.dot(Kpstar, alpha_p) + np.dot(Ksstar, alpha_s)
#        #print 'mean pred={0}'.format(time.time()-t)

  
#        #t = time.time()
#        if ret_var:
#            #latent kernel + priate kernel
#            kstar = kernel.cov(X, task=0, latent=True, diag=True) + kernel.cov(X, task=0, latent=False, diag=True) 
#            Vp1 = np.linalg.solve(Lp, Kpstar.T)
#            Vp2 = np.zeros((len(X), ms))
#            Rstar = np.dot(np.dot(Kpstar, iKp), Ksp.T)
#            for i in xrange(mtasks):
#                start = itask[i]
#                end = itask[i+1]
#                Vp2[:, start:end] = np.linalg.solve(Ld[start:end, start:end], Rstar[:, start:end].T).T
#
#            diagVp = np.sum(Vp1*Vp1, 0) + np.sum(Vp2*Vp2,1)
#    
#            Vs = np.zeros((len(X),ms))
#            diagVs = np.zeros(len(X))
#            
#            for i in xrange(mtasks):
#                start = itask[i]
#                end = itask[i+1]
#                Vs[:,start:end] = np.linalg.solve(Ld[start:end,start:end], Ksstar[:,start:end].T).T
#                diagVs += np.sum(Vs[:,start:end]*Vs[:,start:end], 1)
#            
#            diagVps = np.diag(np.dot(Rstar, np.dot(iD, Ksstar.T)))
#            var = kstar - diagVp - diagVs + 2*diagVps
#            return yfit, var
#        
#            
#        return yfit



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.io as sio
    import scipy.optimize as spopt
    
    from upgeo.base.kernel import SEKernel, ExpGaussianKernel, NoiseKernel, DiracConvolvedKernel 
    from upgeo.mtl.kernel import ConvolvedMTLKernel
    from upgeo.amtl.util import gendata_fmtlgp_1d
    
    
    #generate test data    
    kernel = SEKernel(np.log(0.5), np.log(1))
    #x, yp, ys = gendata_fmtlgp_1d(kernel, kernel, -5, 5, 10, 1, 0.3, 384728364)
    x, yp, ys = gendata_fmtlgp_1d(kernel, kernel, -5, 5, 150, 1, 0.3, 93287486)
    #x, yp, ys = gendata_fmtlgp_1d(kernel, kernel, -5, 5, 150, 1, 0.3)
    

    #plotting data
    plt.plot(x,yp,'*')
    plt.plot(x,ys[:,0], 'r*')
    plt.show()
    
    #preprocess data
    X = np.repeat(x, 2, 1).T.flatten()
    X = X[:,np.newaxis]
    print X
    y = np.r_[yp,ys.flatten()] 
    X = np.r_[X[0:20], X[60:]]
    y = np.r_[y[0:20], y[60:]]

    #construct mtl kernel
    dpKernel = ExpGaussianKernel(np.log(1))
    #dpKernel = DiracConvolvedKernel(SEKernel(np.log(1),np.log(0.1)))
    #dpKernel = DiracConvolvedKernel(GaussianKernel(np.log(1)))
    idpKernel = SEKernel(np.log(1), np.log(0.1)) + NoiseKernel(np.log(0.1))
    #idpKernel = NoiseKernel(np.log(np.sqrt(0.135335283236613)))
                
    theta = [np.log(1), np.log(1)]
    #theta = [np.log(1)]
    kernel = ConvolvedMTLKernel(dpKernel, theta, 2, idpKernel)    
    
    
    gp = CMOGPRegression(kernel, infer_method=CMOGPOnePassInference)
    gp.fit(X, y, [0, 110])
    target_fct = CMOGPExactInference._TargetFunction(gp)
    params = np.copy(gp.hyperparams)
#    params = np.log(np.abs(np.random.randn(len(params))))
    print 'likel={0}'.format(target_fct(params))    
    print 'gradient={0}'.format(target_fct.gradient(params))
    print 'approx_gradient={0}'.format(spopt.approx_fprime(params, target_fct, np.sqrt(np.finfo(float).eps)))
    
    selector = RandomSubsetSelector(10) 
    gp = SparseCMOGPRegression(kernel, infer_method=SparseCMOGPOnePassInference, selector=selector, approx_type=APPROX_TYPE.PITC, fix_inducing=False)
    gp.fit(X, y, [0, 110])
    target_fct = SparseCMOGPExactInference._TargetFunction(gp)
    params = np.copy(gp.hyperparams)
    params = np.r_[params, gp._Xu.flatten()]
    #print 'params'
    print 'likel_before={0}'.format(gp.log_likel)
    #print 'likel={0}'.format(target_fct(params))
    print 'gradient={0}'.format(target_fct.gradient(params))
    print 'approx_gradient={0}'.format(spopt.approx_fprime(params, target_fct, np.sqrt(np.finfo(float).eps)))
    
    
    #dpKernel = ExpGaussianKernel(np.log(0.1))
    #dpKernel = DiracConvolvedKernel(SEKernel(np.log(1),np.log(0.1)))
    #dpKernel = DiracConvolvedKernel(GaussianKernel(np.log(1)))
    dpKernel = CompoundKernel([ExpGaussianKernel(np.log(0.1)),  ExpGaussianKernel(np.log(0.1))])
    idpKernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(np.sqrt(0.135335283236613)))
    #idpKernel = NoiseKernel(np.log(np.sqrt(0.135335283236613)))
                
    #theta = [np.log(0.1), np.log(1)]
    theta = [np.log(0.1), np.log(1),np.log(0.1), np.log(1)]
    #theta = [np.log(1)]
    kernel = ConvolvedMTLKernel(dpKernel, theta, 2, idpKernel)
    Xu = np.linspace(-5, 5, 15, True)
    Xu = Xu[:,np.newaxis]
    print 'Xu={0}'.format(Xu)
    selector = FixedSelector(Xu)
    gp = SparseCMOGPRegression(kernel, infer_method=SparseCMOGPOnePassInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=True)
    #gp = SparseCMOGPRegression(kernel, beta=100, infer_method=SparseCMOGPOnePassInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=False)
    
    #gp = CMOGPRegression(kernel, infer_method=CMOGPExactInference)
    gp.fit(X, y, [0, 110])
    print 'Ku={0}'.format(gp._Ku)
    print 'Kfu={0}'.format(gp._Kfu)
    print 'Kf1={0}'.format(gp._Kf[0])
    print 'Kf2={0}'.format(gp._Kf[1])
    print 'D1={0}'.format(gp._D[0])
    print 'D2={0}'.format(gp._D[1])
    print 'iD1={0}'.format(gp._iD[0])
    print 'iD2={0}'.format(gp._iD[1])
    print 'A={0}'.format(gp._A)
    print 'iA={0}'.format(gp._iA)
    print 'opt_hyper={0}'.format(np.exp(gp.hyperparams))
    print 'opt_Xu={0}'.format(gp._Xu)
    print 'gradients={0}'.format(gp._likel_fun.gradient())
    Xt = Xt = np.arange(-5,5,0.1)
    Xt = Xt[:,np.newaxis]
    print Xt.shape
    print Xt
    print 'learnlikel={0}'.format(gp.log_likel)
    yfit, var = gp.predict(Xt, ret_var=True)
    print 'yfit={0}'.format(yfit)
    print 'var={0}'.format(var)
    mu, Sigma = gp.posterior(Xt)
    print 'mu={0}'.format(mu)
    print 'Sigma={0}'.format(Sigma)
    print 'predict task'
    print gp.posterior(np.atleast_2d(Xt[0]))
    print gp.predict_task(Xt, 0, ret_var=True)
    print gp.predict_task(Xt, 1, ret_var=True)
    #plot fitted lines
    plt.plot(Xt,yfit[:,0],'b-')
    plt.plot(Xt,yfit[:,1],'r-')
    plt.plot(x,yp,'b*')
    plt.plot(x,ys[:,0], 'r*')
        
    plt.show()
    plt.plot(Xt,yfit[:,0],'b-')
    plt.plot(np.r_[x[0:20],x[60:]],np.r_[yp[0:20], yp[60:]],'b*')
    plt.show()
    
    
    #print yfit
    #print var
    
    data = {'x': x, 'y1': yp, 'y2' : ys}
    sio.savemat('/home/marcel/mogpdata.mat', {'data': data})
    
    Xt = Xt = np.arange(0,6)
    Xt = Xt[:,np.newaxis]
    yfit, var = gp.predict(Xt, ret_var=True)
    print Xt
    print yfit
    print var
    print gp.predict_task(Xt, 0, ret_var=True)
    
    likel_fun = gp.likel_fun
    print 'likel={0}'.format(likel_fun())
    print 'grad={0}'.format(likel_fun.gradient())
    print 'kernel_params={0}'.format(np.exp(gp._kernel.params))
    
    X = np.r_[x[0:20], x[60:]]
    y = np.r_[yp[0:20], yp[60:]]
    gp = GPRegression(SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.1)), infer_method=ExactInference)
    gp.fit(X,y)
    Xt = np.arange(-5,5,0.2)
    Xt = Xt[:,np.newaxis]
    yfit = gp.predict(Xt)
    plt.plot(Xt,yfit,'b-')
    plt.plot(X,y,'b*')
    plt.plot(x[20:60],yp[20:60],'r*')
    plt.show()