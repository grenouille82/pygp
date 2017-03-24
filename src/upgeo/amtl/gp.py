'''
Created on May 2, 2012

@author: marcel
'''
import numpy as np

from scikits.learn.base import BaseEstimator, RegressorMixin
from upgeo.amtl.infer import FocusedOnePassInference, FocusedExactInference,\
    FMOGPOnePassInference, FMOGPExactInference, ApproxType
from upgeo.amtl.likel import SimpleFocusedGaussianLogLikel,\
    FocusedGaussianLogLikel, FMOGPGaussianLogLikel
from scipy.linalg.decomp_cholesky import cho_solve
from upgeo.base.gp import GPRegression
from upgeo.base.infer import ExactInference, OnePassInference
from upgeo.util.metric import mspe
from upgeo.base.likel import GaussianLogLikel
from upgeo.base.kernel import CompoundKernel, ExpARDSEKernel, ARDSEKernel,\
    DiracConvolvedKernel, ConstantKernel, ZeroKernel
from upgeo.amtl.kernel import DiracConvolvedAMTLKernel
from upgeo.base.util import plot1d_gp

class FocusedGPRegression(BaseEstimator, RegressorMixin):
    '''
    @todo: - allow to specify a mean function
           - define priors on hyperparameters
           - make noise term explicit, that means likel_fun should include this term
    '''
    
    __slots__ = ('_init_sKernel',
                 '_init_rho',   
                 
                 '_pKernel',    #covariance function for the primary task
                 '_sKernels',   #vector of covariance functions for the secondary tasks
                 '_mtp',        #number of hyperparameters of primary kernel        
                 '_mts',        #number of hyperparameters of the secondary kernels
                 '_rhos',       #correlation vector between the secondary tasks and primary task
                 
                 '_likel_fun',  #likelihood function of the model
                 '_infer',      #inference algorithm used to optimize hyperparameters
                 
                 '_Xp',         #covariates of training set for the primary task
                 '_yp',         #targets of training set for the primary task
                 '_Xs',         #covariates of the training set for the secondary tasks
                 '_ys',         #targets of the training set for the secondary tasks
                 '_itask',      #tasks indices denotes the starting index of each specific task data in Xs, ys
        
                 
                 '_Kp',         #covariance matrix of the primary task data
                 '_Lp',         #cholesky decomposition of Kp
                 '_Lpy',        #Lp^(-1)*y
                 '_iKp',        #inverse of Kp
                 '_Ksp',        #covariance matrix between data of the secondary tasks and primary task
                 '_Lq',         #Lq = chol(Q) with Q = Ksp*Kp^(-1)*Ksp.T
                 '_diagKs',     #diag covariance matrix between the data of the secondary tasks
                 
                 #block diagonals
                 '_Kpriv',      #vector of blk diag cov matrices between the secondary tasks
                 '_G',          #G = diag(Ks - Ksp*Kp^(-1)*Ksp.T)
                 '_D',          #blk diag cov matrix of the secondary tasks: Kpriv+G
                 '_iD',         #inverse of iD
                 '_Ld',         #cholesky decompposition of D
                 
                 '_rp',         #iKp*yp
                 '_rsp',        #Ksp*iKp*yp
                 '_rs',         #iD*ys
                 '_rdsp',       #iD*Ksp*iKp*yp
                 '_rps',        #Kps*iD*ys
                 '_rips',       #iKp*Kps*iD*ys   
                 
                 
        
                 #'_K',          #covariance matrix of the training set
                 #'_L',          #cholesky decomposition of the covariance matrix
                 '_alpha',      #weight vector for each data point???
                 
                 '_log_likel',  #log likelihood of the trained model
                           
                 '_d',          #dimension of each input vector
                 '_m',          #total number of training samples
                 '_mp',         #number of training samples for the primary tasks
                 '_ms',         #total number of training samples for the secondary task
                 '_mtasks',     #number of secondary tasks
                 '_task_sizes', #array of number for each secondary task         
                 
                 '_is_init')

    def __init__(self, pKernel, sKernel, rho=0.1, likel_fun=FocusedGaussianLogLikel, 
                 infer_method=FocusedOnePassInference(ApproxType.PITC)):
        '''
        Constructor
        '''
            
        self._pKernel = pKernel
        self._sKernels = None
        self._init_sKernel = sKernel
        self._init_rho = rho
        self._mtp = pKernel.n_params
        self._mts = sKernel.n_params
        self._likel_fun = likel_fun(self)
        self._infer = infer_method
        
        self._is_init = False


    def fit(self, Xp, yp, Xs, ys, itask):
        '''
        @todo: parameter check
        '''
        
        self._Xp = Xp
        self._yp = yp
        self._Xs = Xs
        self._ys = ys
        
        mp,d = Xp.shape
        ms = Xs.shape[0]
        m = mp+ms
        mtasks = len(itask)
        
        self._d = d
        self._mp = mp
        self._ms = ms
        self._m = m
        self._mtasks = mtasks
        self._itask = itask
        
        self._task_sizes = np.diff(np.r_[itask, ms])
        
        #initialize the task specific kernels and 
        #the correlation vector to the primary task
        self._rhos = np.ones(mtasks)*self._init_rho
        #hard-coded fpr debugging
        #self._rhos = np.log(np.array([0.1, 1]))
        #if self._sKernels == None:
        #gp = GPRegression(self._pKernel, infer_method=ExactInference)
        #gp.fit(Xp, yp)
        sKernels = np.empty(mtasks, dtype=np.object)
        itask = np.r_[self._itask, ms] #add the total number of sec tasks for easier iteration
        for i in xrange(mtasks):
            start = itask[i]
            end = itask[i+1]
        
            sKernels[i] = self._init_sKernel.copy()
            #gp = GPRegression(sKernels[i], infer_method=ExactInference)
            #gp.fit(Xs[start:end], ys[start:end])
            
        self._sKernels = sKernels
         
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
    
        Xp = self._Xp
        Xs = self._Xs
         
        ms = self._ms
        mtasks = self._mtasks

        iKp = self._iKp
        Ksp = self._Ksp
        Ld = self._Ld
        Lp = self._Lp
        iD = self._iD
    
        alpha_p = self._alpha_p
        alpha_s = self._alpha_s 
                
        itask = np.r_[self._itask, ms] #add the total number of sec tasks for easier iteration
        task_sizes = self._task_sizes
            
        #t = time.time()
        pKernel = self._pKernel
        rhos = np.exp(self._rhos)
        Kpstar = pKernel(X, Xp)
        Ksstar = pKernel(X, Xs)*rhos.repeat(task_sizes)
        #Ksstar = np.dot(Kpstar, Ksp.T)
        
        print 'fuzr'
        print Ksstar.shape
        print rhos.repeat(task_sizes).shape
        print pKernel(X, Xs)
        print pKernel(X, Xs)*rhos.repeat(task_sizes)
        print'rho={0}'.format(rhos.repeat(task_sizes))
        print 'pKernel={0}'.format(np.exp(pKernel.params))
        print 'sKernel={0}'.format(np.exp(self._sKernels[0].params))
        
        nfKp = pKernel(Xp, Xp)
        nfiKp = np.linalg.inv(nfKp + np.eye(self._mp)*1e-6)
        
        mp = self._mp
        m = self._m
        Kp = self._Kp
        Kpriv = self._Kpriv
        G = self._G
        #build for simplicity and debugging the full covariance matrix
        #Ks = pKernel(Xs, Xs) * np.repeat(np.exp(2*self._rhos), self._task_sizes)        
        K = np.empty((m,m))
        K[0:mp, 0:mp] = Kp
        K[mp:m, 0:mp] = Ksp
        K[0:mp, mp:m] = Ksp.T
        if self._infer.approx_type == ApproxType.FITC:
            K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfKp),Ksp.T) + np.diag(G) + Kpriv
        else:
            print 'spong'
            K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfKp),Ksp.T) + G + Kpriv
            
        Ks = self._pKernel(Xs,Xs)
        Ks = Ks * np.repeat(rhos,self._task_sizes)
        Ks = (Ks.T * np.repeat(rhos,self._task_sizes)).T
        K[mp:m, mp:m] = Ks + Kpriv
        
        
        iK = np.linalg.inv(K)
        y = np.r_[self._yp, self._ys]
        
        print 'prediction'
        #yfit = (np.dot(Kpstar, alpha_p) + np.dot(Ksstar, alpha_s))
        #print yfit
        yfit = np.dot(np.dot(np.hstack((Kpstar, Ksstar)), iK),y)
        #print yfit 
        #print 'mean pred={0}'.format(time.time()-t)  
        #t = time.time()
        if ret_var:
            kstar = pKernel(X, diag=True)
            Vp1 = np.linalg.solve(Lp, Kpstar.T)
            Vp2 = np.zeros((len(X), ms))
            Rstar = np.dot(np.dot(Kpstar, iKp), Ksp.T)
            for i in xrange(mtasks):
                start = itask[i]
                end = itask[i+1]
                Vp2[:, start:end] = np.linalg.solve(Ld[start:end, start:end], Rstar[:, start:end].T).T

            diagVp = np.sum(Vp1*Vp1, 0) + np.sum(Vp2*Vp2,1)
    
            Vs = np.zeros((len(X),ms))
            diagVs = np.zeros(len(X))
            
            for i in xrange(mtasks):
                start = itask[i]
                end = itask[i+1]
                Vs[:,start:end] = np.linalg.solve(Ld[start:end,start:end], Ksstar[:,start:end].T).T
                diagVs += np.sum(Vs[:,start:end]*Vs[:,start:end], 1)
            
            diagVps = np.diag(np.dot(Rstar, np.dot(iD, Ksstar.T)))
            var = kstar - diagVp - diagVs + 2*diagVps
            return yfit, var
        
            
        return yfit
        
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        pKernel = self._pKernel
        sKernels = self._sKernels
        mtasks = self._mtasks
        
        theta_p = pKernel.params
        theta_s = np.empty(0)
        for i in xrange(mtasks):
            theta_s = np.r_[theta_s, sKernels[i].params]
        rhos = self._rhos
        return np.r_[theta_p, theta_s, rhos]
    
    def _set_hyperparams(self, params):
        '''
        '''
        pKernel = self._pKernel
        sKernels = self._sKernels
        mtp = self._mtp
        mts = self._mts
        mtasks = self._mtasks
        
        theta_p = params[:mtp]
        theta_s = params[mtp:mtp+mtasks*mts]
        theta_s = np.reshape(theta_s, (mtasks, mts))
        rhos = params[mtp+mtasks*mts:]
        
        pKernel.params = theta_p
        for i in xrange(mtasks):
            sKernels[i].params = theta_s[i]
        self._rhos = rhos

    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)
        
class FMOGPRegression(BaseEstimator, RegressorMixin):
    '''
    @todo: - allow to specify a mean function
           - define priors on hyperparameters
           - make noise term explicit, that means likel_fun should include this term
           
    '''
    
    __slots__ = (
                 '_kernel',    #covariance function is AMTLKernel 
                 '_mtp',        #number of hyperparameters of primary kernel        
                 '_mts',        #number of hyperparameters of the secondary kernels
                 
                 '_likel_fun',  #likelihood function of the model
                 '_infer',      #inference algorithm used to optimize hyperparameters
                 
                 '_Xp',         #covariates of training set for the primary task
                 '_yp',         #targets of training set for the primary task
                 '_Xs',         #covariates of the training set for the secondary tasks
                 '_ys',         #targets of the training set for the secondary tasks
                 '_itask',      #tasks indices denotes the starting index of each specific task data in Xs, ys
        
                 
                 '_Kp',         #covariance matrix of the primary task data
                 '_Lp',         #cholesky decomposition of Kp
                 '_Lpy',        #Lp^(-1)*y
                 '_iKp',        #inverse of Kp
                 '_Ksp',        #covariance matrix between data of the secondary tasks and primary task
                 '_Lq',         #Lq = chol(Q) with Q = Ksp*Kp^(-1)*Ksp.T
                 '_diagKs',     #diag covariance matrix between the data of the secondary tasks
                 
                 #block diagonals
                 '_Kpriv',      #vector of blk diag cov matrices between the secondary tasks
                 '_G',          #G = diag(Ks - Ksp*Kp^(-1)*Ksp.T)
                 '_D',          #blk diag cov matrix of the secondary tasks: Kpriv+G
                 '_iD',         #inverse of iD
                 '_Ld',         #cholesky decompposition of D
                 
                 '_rp',         #iKp*yp
                 '_rsp',        #Ksp*iKp*yp
                 '_rs',         #iD*ys
                 '_rdsp',       #iD*Ksp*iKp*yp
                 '_rps',        #Kps*iD*ys
                 '_rips',       #iKp*Kps*iD*ys   
                 
                 
        
                 #'_K',          #covariance matrix of the training set
                 #'_L',          #cholesky decomposition of the covariance matrix
                 '_alpha',      #weight vector for each data point???
                 
                 '_log_likel',  #log likelihood of the trained model
                           
                 '_d',          #dimension of each input vector
                 '_m',          #total number of training samples
                 '_mp',         #number of training samples for the primary tasks
                 '_ms',         #total number of training samples for the secondary task
                 '_mtasks',     #number of secondary tasks
                 '_task_sizes', #array of number for each secondary task         
                 
                 '_is_init')

    def __init__(self, kernel, likel_fun=FMOGPGaussianLogLikel,  
                 infer_method=FMOGPOnePassInference):
        '''
        Constructor
        '''
            
        self._kernel = kernel
        #self._mtp = pKernel.n_params
        #self._mts = sKernel.n_params
        self._likel_fun = likel_fun(self)
        self._infer = infer_method()
        
        self._is_init = False


    def fit(self, Xp, yp, Xs, ys, itask):
        '''
        @todo: parameter check
        '''
        
        self._Xp = Xp
        self._yp = yp
        self._Xs = Xs
        self._ys = ys
        
        mp,d = Xp.shape
        ms = Xs.shape[0]
        m = mp+ms
        mtasks = len(itask)
        
        self._d = d
        self._mp = mp
        self._ms = ms
        self._m = m
        self._mtasks = mtasks
        self._itask = itask
        
        self._task_sizes = np.diff(np.r_[itask, ms])
        
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
        n = len(X)
    
        Xp = self._Xp
        Xs = self._Xs
         
        ms = self._ms
        mtasks = self._mtasks

        iKp = self._iKp
        Ksp = self._Ksp
        Ld = self._Ld
        Lp = self._Lp
        iD = self._iD
    
        alpha_p = self._alpha_p
        alpha_s = self._alpha_s 
                
        itask = np.r_[self._itask, ms] #add the total number of sec tasks for easier iteration
        task_sizes = self._task_sizes
            
        #t = time.time()
        kernel = self._kernel
        #latent kernel + priate kernel
        Kpstar = kernel.cross_cov(X, Xp, task=0, latent=True) + kernel.cross_cov(X, Xp, task=0, latent=False)
        Ksstar = np.zeros((n, ms))
        for i in xrange(mtasks):
            start = itask[i]
            end = itask[i+1]
            Ksstar[:,start:end] = kernel.cross_cov(X, Xs[start:end], task=i+1, latent=True)
            
        yfit = np.dot(Kpstar, alpha_p) + np.dot(Ksstar, alpha_s)
        #print 'mean pred={0}'.format(time.time()-t)  
        #t = time.time()
        if ret_var:
            #latent kernel + priate kernel
            kstar = kernel.cov(X, task=0, latent=True, diag=True) + kernel.cov(X, task=0, latent=False, diag=True) 
            Vp1 = np.linalg.solve(Lp, Kpstar.T)
            Vp2 = np.zeros((len(X), ms))
            Rstar = np.dot(np.dot(Kpstar, iKp), Ksp.T)
            for i in xrange(mtasks):
                start = itask[i]
                end = itask[i+1]
                Vp2[:, start:end] = np.linalg.solve(Ld[start:end, start:end], Rstar[:, start:end].T).T

            diagVp = np.sum(Vp1*Vp1, 0) + np.sum(Vp2*Vp2,1)
    
            Vs = np.zeros((len(X),ms))
            diagVs = np.zeros(len(X))
            
            for i in xrange(mtasks):
                start = itask[i]
                end = itask[i+1]
                Vs[:,start:end] = np.linalg.solve(Ld[start:end,start:end], Ksstar[:,start:end].T).T
                diagVs += np.sum(Vs[:,start:end]*Vs[:,start:end], 1)
            
            diagVps = np.diag(np.dot(Rstar, np.dot(iD, Ksstar.T)))
            var = kstar - diagVp - diagVs + 2*diagVps
            return yfit, var
        
            
        return yfit
    
class AMOGPRegression(BaseEstimator, RegressorMixin):
    '''
    @todo: - allow to specify a mean function
           - define priors on hyperparameters
           - make noise term explicit, that means likel_fun should include this term
    '''
    
    __slots__ = ('_init_sKernel',
                 '_init_rho',   
                 
                 '_pKernel',    #covariance function for the primary task
                 '_sKernels',   #vector of covariance functions for the secondary tasks
                 '_mtp',        #number of hyperparameters of primary kernel        
                 '_mts',        #number of hyperparameters of the secondary kernels
                 '_rhos',       #correlation vector between the secondary tasks and primary task
                 
                 '_likel_fun',  #likelihood function of the model
                 '_infer',      #inference algorithm used to optimize hyperparameters
                 
                 '_Xu',
                 '_Xp',         #covariates of training set for the primary task
                 '_yp',         #targets of training set for the primary task
                 '_Xs',         #covariates of the training set for the secondary tasks
                 '_ys',         #targets of the training set for the secondary tasks
                 '_itask',      #tasks indices denotes the starting index of each specific task data in Xs, ys
        
                 
                 '_Kp',         #covariance matrix of the primary task data
                 '_Lp',         #cholesky decomposition of Kp
                 '_Lpy',        #Lp^(-1)*y
                 '_iKp',        #inverse of Kp
                 '_Ksp',        #covariance matrix between data of the secondary tasks and primary task
                 '_Lq',         #Lq = chol(Q) with Q = Ksp*Kp^(-1)*Ksp.T
                 '_diagKs',     #diag covariance matrix between the data of the secondary tasks
                 
                 #block diagonals
                 '_Kpriv',      #vector of blk diag cov matrices between the secondary tasks
                 '_G',          #G = diag(Ks - Ksp*Kp^(-1)*Ksp.T)
                 '_D',          #blk diag cov matrix of the secondary tasks: Kpriv+G
                 '_iD',         #inverse of iD
                 '_Ld',         #cholesky decompposition of D
                 
                 '_rp',         #iKp*yp
                 '_rsp',        #Ksp*iKp*yp
                 '_rs',         #iD*ys
                 '_rdsp',       #iD*Ksp*iKp*yp
                 '_rps',        #Kps*iD*ys
                 '_rips',       #iKp*Kps*iD*ys   
                 
                 
        
                 #'_K',          #covariance matrix of the training set
                 #'_L',          #cholesky decomposition of the covariance matrix
                 '_alpha',      #weight vector for each data point???
                 
                 '_log_likel',  #log likelihood of the trained model
                           
                 '_d',          #dimension of each input vector
                 '_m',          #total number of training samples
                 '_mp',         #number of training samples for the primary tasks
                 '_ms',         #total number of training samples for the secondary task
                 '_mtasks',     #number of secondary tasks
                 '_task_sizes', #array of number for each secondary task         
                 
                 '_is_init')

    def __init__(self, pKernel, sKernel, rho=0.1, likel_fun=FocusedGaussianLogLikel, 
                 infer_method=FocusedOnePassInference(ApproxType.PITC)):
        '''
        Constructor
        '''
            
        self._pKernel = pKernel
        self._sKernels = None
        self._init_sKernel = sKernel
        self._init_rho = rho
        self._mtp = pKernel.n_params
        self._mts = sKernel.n_params
        self._likel_fun = likel_fun(self)
        self._infer = infer_method
        
        self._is_init = False


    def fit(self, Xp, yp, Xs, ys, itask):
        '''
        @todo: parameter check
        '''
        if Xp.shape[1] > 1:
            perm = np.random.permutation(np.arange(0,len(Xp)))
            Xu = Xp[perm[0:30]]
        else:
            Xu = np.linspace(np.min(Xp[:,0]), np.max(Xp[:,0]), 30)
            Xu = Xu[:,np.newaxis]
            
        self._Xu = Xu
        
        print 'Xu'
        print Xu
        
        self._Xp = Xp
        self._yp = yp
        self._Xs = Xs
        self._ys = ys
        
        mp,d = Xp.shape
        ms = Xs.shape[0]
        m = mp+ms
        mtasks = len(itask)
        
        self._d = d
        self._mp = mp
        self._ms = ms
        self._m = m
        self._mtasks = mtasks
        self._itask = itask
        
        self._task_sizes = np.diff(np.r_[itask, ms])
        
        #initialize the task specific kernels and 
        #the correlation vector to the primary task
        self._rhos = np.ones(mtasks)*self._init_rho
        #hard-coded fpr debugging
        #self._rhos = np.log(np.array([0.1, 1]))
        #if self._sKernels == None:
        #gp = GPRegression(self._pKernel, infer_method=ExactInference)
        #gp.fit(Xp, yp)
        sKernels = np.empty(mtasks, dtype=np.object)
        itask = np.r_[self._itask, ms] #add the total number of sec tasks for easier iteration
        for i in xrange(mtasks):
            start = itask[i]
            end = itask[i+1]
        
            sKernels[i] = self._init_sKernel.copy()
            #gp = GPRegression(sKernels[i], infer_method=ExactInference)
            #gp.fit(Xs[start:end], ys[start:end])
            
        self._sKernels = sKernels
         
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
    
        Xp = self._Xp
        Xs = self._Xs
         
        ms = self._ms
        mtasks = self._mtasks

        iKp = self._iKp
        Ksp = self._Ksp
        Ld = self._Ld
        Lp = self._Lp
        iD = self._iD
    
        alpha_p = self._alpha_p
        alpha_s = self._alpha_s 
                
        itask = np.r_[self._itask, ms] #add the total number of sec tasks for easier iteration
        task_sizes = self._task_sizes
       
        mp = self._mp
        m = self._m
        Xu = self._Xu
       
            
        #t = time.time()
        pKernel = self._pKernel
        rhos = np.exp(self._rhos)
        Kpstar = pKernel(X, Xp)
        Ksstar = pKernel(X, Xu)#*rhos.repeat(task_sizes)
        #Ksstar = np.dot(Kpstar, Ksp.T)
        
        print 'fuzr'
        print Ksstar.shape
        print rhos.repeat(task_sizes).shape
        print pKernel(X, Xs)
        print pKernel(X, Xs)*rhos.repeat(task_sizes)
        print'rho={0}'.format(rhos.repeat(task_sizes))
        print 'pKernel={0}'.format(np.exp(pKernel.params))
        print 'sKernel={0}'.format(np.exp(self._sKernels[0].params))
        
        
        
        Ku = pKernel(Xu,Xu)
        Ksu = pKernel(Xs, Xu)
        Ksu = (Ksu.T * np.repeat(rhos, task_sizes)).T
        Kpu = pKernel(Xp, Xu)
        iKu = np.linalg.inv(Ku + np.eye(len(Xu))*1e-6)
        Kp = self._Kp
        Ksp = np.dot(np.dot(Ksu, iKu), Kpu.T)
        
        Ks = np.dot(np.dot(Ksu, iKu), Ksu.T)
        G = np.diag(pKernel(Xs,Xs))*np.repeat(rhos**2,self._task_sizes)-np.diag(Ks)
        
        Kpriv = self._Kpriv
        K = np.zeros((m,m))
        K[0:mp, 0:mp] = Kp
        K[mp:m, 0:mp] = Ksp
        K[0:mp, mp:m] = Ksp.T
        
        K[mp:m, mp:m] = Ks + np.diag(G) + Kpriv
        
        iK = np.linalg.inv(K)
        y = np.r_[self._yp, self._ys]
        Ksstar = np.dot(np.dot(Ksstar,iKu), Ksu.T)
        print 'prediction'
        #yfit = (np.dot(Kpstar, alpha_p) + np.dot(Ksstar, alpha_s))
        #print yfit
        yfit = np.dot(np.dot(np.hstack((Kpstar, Ksstar)), iK),y)
        #print yfit 
        #print 'mean pred={0}'.format(time.time()-t)  
        #t = time.time()
        if ret_var:
            kstar = pKernel(X, diag=True)
            Vp1 = np.linalg.solve(Lp, Kpstar.T)
            Vp2 = np.zeros((len(X), ms))
            Rstar = np.dot(np.dot(Kpstar, iKp), Ksp.T)
            for i in xrange(mtasks):
                start = itask[i]
                end = itask[i+1]
                Vp2[:, start:end] = np.linalg.solve(Ld[start:end, start:end], Rstar[:, start:end].T).T

            diagVp = np.sum(Vp1*Vp1, 0) + np.sum(Vp2*Vp2,1)
    
            Vs = np.zeros((len(X),ms))
            diagVs = np.zeros(len(X))
            
            for i in xrange(mtasks):
                start = itask[i]
                end = itask[i+1]
                Vs[:,start:end] = np.linalg.solve(Ld[start:end,start:end], Ksstar[:,start:end].T).T
                diagVs += np.sum(Vs[:,start:end]*Vs[:,start:end], 1)
            
            diagVps = np.diag(np.dot(Rstar, np.dot(iD, Ksstar.T)))
            var = kstar - diagVp - diagVs + 2*diagVps
            return yfit, var
        
            
        return yfit
        
    def _get_hyperparams(self):
        '''
        @todo: eventually return a copy
        '''
        pKernel = self._pKernel
        sKernels = self._sKernels
        mtasks = self._mtasks
        
        theta_p = pKernel.params
        theta_s = np.empty(0)
        for i in xrange(mtasks):
            theta_s = np.r_[theta_s, sKernels[i].params]
        rhos = self._rhos
        return np.r_[theta_p, theta_s, rhos]
    
    def _set_hyperparams(self, params):
        '''
        '''
        pKernel = self._pKernel
        sKernels = self._sKernels
        mtp = self._mtp
        mts = self._mts
        mtasks = self._mtasks
        
        theta_p = params[:mtp]
        theta_s = params[mtp:mtp+mtasks*mts]
        theta_s = np.reshape(theta_s, (mtasks, mts))
        rhos = params[mtp+mtasks*mts:]
        
        pKernel.params = theta_p
        for i in xrange(mtasks):
            sKernels[i].params = theta_s[i]
        gp._rhos = rhos

    hyperparams = property(fget=_get_hyperparams, fset=_set_hyperparams)


class STLGPRegression(BaseEstimator, RegressorMixin):
    '''
    Wrapper class by using basic GP in a MTL setting by using only training data of the
    primary task
    '''
    __slots__ = ('_gp')
    
    def __init__(self, kernel, likel_fun=GaussianLogLikel, 
                 infer_method=OnePassInference, priors=None):
        self._gp = GPRegression(kernel, likel_fun, infer_method, priors)
    
    def fit(self, Xp, yp, Xs, ys, itask):
        '''
        '''
        self._gp.fit(Xp, yp)
    
    def predict(self, X, ret_var=False):
        '''
        '''
        return self._gp.predict(X, ret_var)
        
       
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
    
class PooledGPRegression(BaseEstimator, RegressorMixin):
    '''
    Wrapper by using basic GP in a MTL setting by pooling the training data of the
    primary task and background tasks.
    '''
    __slots__ = ('_gp')
    
    def __init__(self, kernel, likel_fun=GaussianLogLikel, 
                 infer_method=OnePassInference, priors=None):
        self._gp = GPRegression(kernel, likel_fun, infer_method, priors)
    
    def fit(self, Xp, yp, Xs, ys, itask):
        '''
        '''
        X = np.vstack((Xp, Xs))
        y = np.vstack((yp, ys))
        self._gp.fit(X, y)
    
    def predict(self, X, ret_var=False):
        '''
        '''
        return self._gp.predict(X, ret_var)
        
       
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


if __name__ == '__main__':
    from upgeo.base.kernel import SEKernel, NoiseKernel
    from upgeo.amtl.util import gendata_fmtlgp_1d
    
    import time
    import matplotlib.pyplot as plt
    
    pKernel = SEKernel(np.log(0.5), np.log(0.1)) + NoiseKernel(np.log(0.1))
    sKernel = SEKernel(np.log(0.5), np.log(0.1)) + NoiseKernel(np.log(0.1))
    #sKernel = ZeroKernel()
    rho = np.log(0.1)
    
    kernel = SEKernel(np.log(0.5), np.log(1))
    ntasks = 1
    x, yp, ys = gendata_fmtlgp_1d(kernel, kernel, -5, 5, 100, ntasks, sigma=0.1, seed=3932878736)
    print ys.shape
    plt.plot(x,yp,'*')
    plt.show()
    for i in xrange(ntasks):
        plt.plot(x,ys[:,i], '*')
        plt.show()
    #for i in xrange(ntasks):
    #    plt.plot(x,ys[:,i])
    #    plt.show()
    yo = np.copy(yp)
    yss = ys
    
    #preprocess data
    Xp = np.r_[x[0:10], x[80:100]]
    yp = np.r_[yp[0:10], yp[80:100]]
    #Xs = np.tile(x, (ntasks,1))
    Xs = np.tile(x, (ntasks+1,1))
    #ys = ys.ravel()
    ys = np.r_[ys.ravel(), yo]
    itask =  np.arange(0,200,100)
    gp = FocusedGPRegression(pKernel, sKernel, rho, likel_fun=SimpleFocusedGaussianLogLikel)
    gp.fit(Xp, yp, Xs, ys, itask)
    print 'likel={0}'.format(gp._log_likel)
    yfit = gp.predict(x[10:80])
    print yfit
    print np.mean((yfit-yo[10:80])**2)
    plt.plot(x,yo, '*')
    plt.plot(x, gp.predict(x))
    plt.show()
    
    
    print 'halle galle'
    
    
    gp1 = GPRegression(gp._sKernels[0])
    gp1.fit(x, yss[:,0])
    plt.plot(x,yss[:,0], '*')
    plt.plot(x, gp1.predict(x))
    plt.show()
    
    gp1 = GPRegression(gp._sKernels[1])
    gp1.fit(x, yo)
    plt.plot(x,yo, '*')
    plt.plot(x, gp1.predict(x))
    plt.show()
    
    gp1 = GPRegression(gp._pKernel)
    gp1.fit(Xp, yp)
    plt.plot(x,yo, '*')
    plt.plot(x, gp1.predict(x))
    plt.show()
        
    gp = FocusedGPRegression(pKernel, sKernel, rho, likel_fun=FocusedGaussianLogLikel, infer_method=FocusedExactInference)
    t = time.time()
    gp.fit(Xp, yp, Xs, ys, itask)
    print 'fittime={0}'.format(time.time()-t)
    print 'likel={0}'.format(gp._log_likel)
    yfit = gp.predict(x[10:80])
    print yfit
    print 'mean'
    print mspe(yfit, yo[10:80])
    t = time.time()
    gp._likel_fun.gradient()
    print 'gradtime={0}'.format(time.time()-t)
    print gp.hyperparams
    print 'rhos'
    print np.exp(gp._rhos)
    print gp._log_likel
    print 'hyperhype233223'
    print np.exp(gp.hyperparams)
    
    print 'shing'
    plt.plot(x,yo, '*')
    plt.plot(x, gp.predict(x))
    plt.show()
    
    print 'hasso lassa'
    
    gp1 = GPRegression(gp._sKernels[0])
    gp1.fit(x, yss[:,0])
    plt.plot(x,yss[:,0], '*')
    plt.plot(x, gp1.predict(x))
    plt.show()
    
    gp1 = GPRegression(gp._sKernels[1])
    gp1.fit(x, yo)
    plt.plot(x,yo, '*')
    plt.plot(x, gp1.predict(x))
    plt.show()
    
    gp1 = GPRegression(gp._pKernel)
    gp1.fit(Xp, yp)
    plt.plot(x,yo, '*')
    plt.plot(x, gp1.predict(x))
    plt.show()

    gp._rhos = np.log(np.array([0.000001, 1]))
    print 'rhos badman'
    print gp._rhos
    gp._infer = FocusedOnePassInference()
    #gp._likel_fun = SimpleFocusedGaussianLogLikel(gp)
    gp.fit(Xp, yp, Xs, ys, itask)
    print 'rhos salvation'
    print gp._rhos
    print gp._log_likel
    plt.plot(x,yo, '*')
    plt.plot(x, gp.predict(x))
    plt.show()
    print 'hyperhype'
    print np.exp(gp.hyperparams)
    
    print 'furz purz'
    
    #secondory kernels print
    gp1 = GPRegression(gp._sKernels[0])
    gp1.fit(x, yss[:,0])
    plt.plot(x,yss[:,0], '*')
    plt.plot(x, gp1.predict(x))
    plt.show()
    
    gp1 = GPRegression(gp._sKernels[1])
    gp1.fit(x, yo)
    plt.plot(x,yo, '*')
    plt.plot(x, gp1.predict(x))
    plt.show()
    
    gp1 = GPRegression(gp._pKernel)
    gp1.fit(Xp, yp)
    plt.plot(x,yo, '*')
    plt.plot(x, gp1.predict(x))
    plt.show()
    
    #pKernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
    pKernel = SEKernel(np.log(0.1), np.log(1)) + NoiseKernel(np.log(0.5))
    gp = GPRegression(pKernel, infer_method=ExactInference)
    gp.fit(Xp, yp)
    t = time.time()
    gp.predict(x, False)
    print 'predtime={0}'.format(time.time()-t)
    yfit = gp.predict(x[10:80])
    print 'mean'
    print mspe(yfit, yo[10:80])
    print yfit
    print yo[10:80]
    print 'likel'
    print gp.log_likel
    print 'spppppppppppppppppppppppppppaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    plot1d_gp(gp, -5, 5)
    
    plt.plot(x,yo, '*')
    plt.plot(x, gp.predict(x))
    plt.show()
    
    lKernel = CompoundKernel([ExpARDSEKernel(np.log(1), np.log(1))])
    pKernel = ARDSEKernel(np.log(1), np.log(1))+ NoiseKernel(np.log(0.5))
    sKernel = ARDSEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
    
    theta = np.r_[np.log(1), np.log(0.1)]
    mtlKernel = DiracConvolvedAMTLKernel(lKernel, theta, ntasks+1, pKernel, sKernel)
    mtlKernel._theta[0] = np.log(np.array([1, 0.0001]))
    mtlKernel._theta[1] = np.log(np.array([1, 1]))
    
    gp = FMOGPRegression(mtlKernel, infer_method=FMOGPOnePassInference)
    gp.fit(Xp, yp, Xs, ys, itask)
    print 'fittime={0}'.format(time.time()-t)
    print 'likel={0}'.format(gp._log_likel)
    yfit = gp.predict(x[10:80])
    print yfit
    print 'mean'
    print mspe(yfit, yo[10:80])
    plt.plot(x,yo, '*')
    plt.plot(x, gp.predict(x))
    plt.show()
    
    gp = FMOGPRegression(mtlKernel, infer_method=FMOGPExactInference)
    gp.fit(Xp, yp, Xs, ys, itask)
    print 'fittime={0}'.format(time.time()-t)
    print 'likel={0}'.format(gp._log_likel)
    yfit = gp.predict(x[10:80])
    print yfit
    print 'mean'
    print mspe(yfit, yo[10:80])
    plt.plot(x,yo,'*')
    plt.plot(x, gp.predict(x))
    plt.show()

    print pKernel.params
    print lKernel.params
    print mtlKernel.params
    
    lKernel = CompoundKernel([DiracConvolvedKernel(ARDSEKernel(np.log(0.5), np.log(0.1)))])
    #pKernel = ARDSEKernel(np.log(0.5), np.log(0.1))+ NoiseKernel(np.log(0.1))
    pKernel = NoiseKernel(np.log(0.1))
    sKernel = ARDSEKernel(np.log(0.5), np.log(0.1)) + NoiseKernel(np.log(0.1))
    
    theta = np.r_[np.log(1)]
    mtlKernel = DiracConvolvedAMTLKernel(lKernel, theta, ntasks+1, pKernel, sKernel)
    mtlKernel._theta[0] = np.log(np.array([0.0001]))
    mtlKernel._theta[1] = np.log(np.array([1]))
    gp = FMOGPRegression(mtlKernel, infer_method=FMOGPOnePassInference)
    gp.fit(Xp, yp, Xs, ys, itask)
    print 'spam'
    print 'fittime={0}'.format(time.time()-t)
    print 'likel={0}'.format(gp._log_likel)
    yfit = gp.predict(x[10:80])
    print yfit
    print 'mean'
    print mspe(yfit, yo[10:80])
    plt.plot(x,yo, '*')
    plt.plot(x, gp.predict(x))
    plt.show()
    
    gp = FMOGPRegression(mtlKernel, infer_method=FMOGPExactInference)
    gp.fit(Xp, yp, Xs, ys, itask)
    print 'fittime={0}'.format(time.time()-t)
    print 'likel={0}'.format(gp._log_likel)
    yfit = gp.predict(x[10:80])
    print yfit
    print 'mean'
    print mspe(yfit, yo[10:80])
    plt.plot(x,yo,'*')
    plt.plot(x, gp.predict(x))
    plt.show()

    print pKernel.params
    print lKernel.params
    print mtlKernel.params
    
    