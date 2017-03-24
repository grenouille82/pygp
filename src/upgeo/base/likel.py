'''
Created on Aug 4, 2011

@author: marcel
'''

import time
import numpy as np

from abc import ABCMeta, abstractmethod
from scipy.linalg.decomp_cholesky import cho_solve
from upgeo.base.mean import LinearMean, BiasedLinearMean, ConstantMean,\
    HiddenMean

class LikelihoodFunction(object):
    
    __metaclass__ = ABCMeta
    
    __slots__ = ( '_gp'    #the gp model for which the likelihood should be determined
                )
    
    def __init__(self, gp_model):
        self._gp = gp_model
        
    @abstractmethod
    def __call__(self):
        pass
    
    @abstractmethod
    def gradient(self):
        pass
    
class GaussianLogLikel(LikelihoodFunction):
    '''
    @todo: - consider to define the input noise implicit in the likelihood function. 
             at the moment input noise can be formulated by the covariance function 
             as hyperparameter. 
    '''
    
    def __init__(self, gp_model):
        LikelihoodFunction.__init__(self, gp_model)
        
    def __call__(self):
        gp = self._gp
        
        meanfct = gp._meanfct
        X = gp._X
        y = gp._y           #targets of the training cases
        L = gp._L           #cholesky decomposition of the covariance matrix
        n = gp._n           #number of training samples
        alpha = gp._alpha   #weight vector for each kernel component???
        
        
        params = gp.hyperparams
        priors = gp._priors
        
        m = np.zeros(len(y))
        if meanfct != None:
            m = meanfct(X)
        likel = (-np.dot((y-m).T, alpha) - n*np.log(2.0*np.pi)) / 2.0 #check, if we net the transpose
        likel -= np.sum(np.log(np.diag(L)))
        
        if priors != None:
            for i in xrange(len(params)):
                pr = priors[i]
                if pr != None:
                    likel += pr.log_pdf(params[i])
                    
        #print 'likel={0}'.format(likel)
        
        return likel
    
    def gradient(self):
        gp = self._gp
        
        #@todo look for a better way to store the proximity
        
        X = gp._X           #covariates of the training cases
        L = gp._L           #cholesky decomposition of the covariance matrix
        n = gp._n           #number of training samples
        alpha = gp._alpha   #weight vector for each kernel component???
        kernel = gp._kernel #used kernel in gp model
        k = kernel.nparams  #number of hyperparameters
        params = gp.hyperparams
        priors = gp._priors
        
        #print 'grad'
        K_inv = cho_solve((L,1), np.eye(n))
        Q = np.outer(alpha, alpha)-K_inv
        
        #determine the gradient of the kernel hyperparameters
        gradK = np.empty(k)
        for i in xrange(k):
            #compute the gradient of the i-th hyperparameter
            d_kernel = kernel.derivate(i)
            dK = d_kernel(X)
            gradK[i] = np.sum(Q*dK) / 2.0
            if priors != None:
                pr = priors[i]
                if pr != None:
                    gradK[i] += pr.log_gradient(params[i])
                    
        #determine the gradients of the mean fct hyperparameters if present
        meanfct = gp._meanfct
        if meanfct != None:
            offset = k #need to determine the correct prior
            k = meanfct.nparams
            gradM = np.empty(k)
            for i in xrange(k):
                d_meanfct = meanfct.derivate(i)
                dM = d_meanfct(X)
                gradM[i] = np.dot(dM, alpha) #check if the gradient is correct
                if priors != None:
                    pr = priors[offset+i]
                    gradM[i] += pr.log_gradient(params[i])
            grad = np.r_[gradK, gradM]
        else:
            grad = gradK
                    
        return grad
    
class SparseGaussianLogLikel(LikelihoodFunction):

    def __init__(self, gp_model):
        LikelihoodFunction.__init__(self, gp_model)
        
    def __call__(self):
        '''
        @todo: - optimize the quadratic forms
        '''
        gp = self._gp
        
        meanfct = gp._meanfct
        
        n = gp._n
        X = gp._X
        y = gp._y
        
        Lnm = gp._Lnm
        V = gp._V
        G = gp._G       
        
        my = np.zeros(len(y))
        if meanfct != None:
            my = meanfct(X)

        
        #1.part of the woodbury inverse (y*G*y)
        w1 = (y-my)/np.sqrt(G)
        #2.part of the woodbury inverse 
        w2 = np.linalg.solve(Lnm, np.dot(V,w1))
        
        l1 = np.dot(w1,w1) - np.dot(w2,w2)
        l2 = np.sum(np.log(G)) + 2.0*np.sum(np.log(np.diag(Lnm)))
        
        likel = (-l1 - l2 - n*np.log(2.0*np.pi)) / 2.0
        return likel
    
    def gradient(self):
        '''
        @todo: - mark as depricated because this method is slower than gradientFast and 
                 this method has not the ability to optimize Xu in a direct way
        '''
        gp = self._gp
        X = gp._X
        y = gp._y
        Xu = gp._Xu
        
        kernel = gp._kernel
        k = kernel.nparams
        
        Knm = gp._Knm
        
        iQ = gp._iQ
        iKm = gp._iKm
        G = gp._G
        
        uKnm = Knm.T/np.sqrt(G)
        uKnm = uKnm.T
        
        #t = time.time()
        B1 = np.dot(iKm, Knm.T)
        B2 = np.dot(iQ, uKnm.T)
        r = y/np.sqrt(G)
        Br = np.dot(B2,r)
  
        #print 'B1={0}'.format(time.time()-t)
        
        grad = np.empty(k)
        for i in xrange(k):
            #todo: optimize the quadratic terms
            dkernel = kernel.derivate(i)
            
            dKm = dkernel(Xu, Xu)
            dKn = dkernel(X, diag=True)
            dKnm = dkernel(X, Xu)
            
            
            
            duKnm = dKnm.T/np.sqrt(G)
            duKnm = duKnm.T
            
            
           
            #compute diag(dQn)
            #B = np.dot(iKm, Knm.T)
            R = 2*dKnm.T - np.dot(dKm, B1)
            ddQn = np.sum(R*B1,0)
            
            
            #compute gradient dG
            dG =  dKn - ddQn
            
            qfG = dG/(G**2)
            V = np.dot(duKnm.T, uKnm)
            dQ = dKm + V+V.T - np.dot(Knm.T*qfG, Knm)
           
            #B = np.dot(iQ, uKnm.T)
            #r = y/np.sqrt(G)
            #Br = np.dot(B,r)
            
            
            V = -2.0*Knm.T*qfG + 2.0*dKnm.T/G 
            V -= np.dot(B2.T, dQ).T/np.sqrt(G)
            V = np.dot(V.T, Br)  
            
            dl1 = -np.dot(y*qfG, y) - np.dot(y, V)
            dl2 = np.sum(1.0/G*dG) - np.sum(iKm*dKm) + np.sum(iQ*dQ)
            
            grad[i] = (-dl1-dl2) / 2.0
            
        return grad
    
    def gradientFast(self):
        '''
        Very Fast gradient computation of the kernel hyperparameters and inducing data points 
        by using chain rule of the likelihood derivative (see Alvarez 2011).
        
        '''
        gp = self._gp
        
        meanfct = gp._meanfct
        
        #data 
        X = gp._X
        y = gp._y
        Xu = gp._Xu
        
        my = np.zeros(len(y))
        if meanfct != None:
            my = meanfct(X)
        
        #precomuted stuff
        iKm = gp._iKm
        Knm = gp._Knm
        G = gp._G
        alpha = gp._alpha
        iQ = gp._iQ
        kernel = gp._kernel
        noise_kernel = gp._noise_kernel
        
        #formulas uses identical notation to the Paper of Alvarez
        #so the diag matrices are computed in a efficient way
    
        t = time.time()
        yy = (y-my)*(y-my)    
        C = iQ + np.outer(alpha, alpha)
        B3 = np.dot(Knm, alpha)
        H = G - yy + 2.0*B3*(y-my)
        J = H-np.sum(np.dot(Knm, C)*Knm,1)
        Q = J/(G**2)
        
        B1 = np.dot(iKm, Knm.T)
        B2 = B1*Q

        dKn = -0.5*Q
        dKm = 0.5*(iKm - C - np.dot(B2,B1.T))
        dKmn = B2 - np.dot(C, Knm.T)/G + np.outer(alpha,(y-my)/G)
        dKmn = dKmn.T
        #print 'grad precomp={0}'.format(time.time()-t)


        t = time.time()
        k = kernel.nparams
        grad = np.empty(k)
        for i in xrange(k):
            dkernel = kernel.derivate(i)
            
            dTm = dkernel(Xu, Xu)
            dTn = dkernel(X, diag=True)
            dTnm = dkernel(X, Xu)
            grad[i] = np.dot(dTn, dKn) 
            grad[i] += np.dot(dTm.flatten(), dKm.flatten()) 
            grad[i] += np.dot(dTnm.flatten(), dKmn.flatten())
        
        if noise_kernel != None:
            k = noise_kernel.nparams
            gradNK = np.empty(k)
            for i in xrange(k):
                dkernel = noise_kernel.derivate(i)
                dTn = dkernel(X, diag=True)
                gradNK[i] = np.dot(dTn, dKn)
            grad = np.r_[grad, gradNK]
        
        #print 'grad kernel params={0}'.format(time.time()-t)
        
        #determine the gradients of the mean fct hyperparameters if present
        if meanfct != None:
            k = meanfct.nparams
            gradM = np.empty(k)
            dM = (y-my)/G - 1/G*B3
            for i in xrange(k):
                d_meanfct = meanfct.derivate(i)
                dTM = d_meanfct(X)
                gradM[i] = np.dot(dTM, dM) #check if the gradient is correct
            grad = np.r_[grad, gradM]
            

        t = time.time()
        if not gp._fix_inducing:
            t1 = time.time()
            dkernelX = kernel.derivateX()
            dXm = dkernelX(Xu,Xu)*2.0
            dXmn = dkernelX(Xu,X)
            #print 'grad incducing params partI={0}'.format(time.time()-t1)
            m,d = Xu.shape
            gradX = np.zeros((m,d))
            
            t1 = time.time()
            for i in xrange(m):
                for j in xrange(d):
                    gradX[i,j] = np.dot(dXm[:,i,j], dKm[:,i])
                    gradX[i,j] += np.dot(dXmn[:,i,j], dKmn[:,i])
            #print 'grad incducing params partII={0}'.format(time.time()-t1)
            grad = np.r_[grad, gradX.flatten()]
        #print 'grad incducing params={0}'.format(time.time()-t)

        return grad 

class PITCSparseGaussianLogLikel(LikelihoodFunction):

    def __init__(self, gp_model):
        LikelihoodFunction.__init__(self, gp_model)
        
    def __call__(self):
        '''
        @todo: - optimize the quadratic forms
        '''
        gp = self._gp
        
        meanfct = gp._meanfct
        
        n = gp._n
        X = gp._X
        y = gp._y
        
        #Lnm = gp._Lnm
        #V = gp._V
        #G = gp._G
        Lg = gp._Lg
        iGy = gp._iGy
        r = gp._r
        alpha = gp._alpha
        
        my = np.zeros(len(y))
        if meanfct != None:
            my = meanfct(X)

        igroup = gp._igroup
        k = len(igroup)
        igroup = np.r_[igroup, n]
        
        #1.part of the woodbury inverse (y*G*y)
        w1 = np.zeros(n)
        detG = 0
        #iGy = np.zeros(n)
        #iGKnm = np.zeros((n,len(gp._Xu)))
        #Knmy = np.dot(gp._Knm.T,y-my)
        
        for i in xrange(k):
            #start = igroup[i]
            #end = igroup[i+1]
            #w1[start:end] = np.linalg.solve(Lg[i], y[start:end]-my[start:end])
            detG += np.sum(np.log(np.diag(Lg[i])))
        
        #2.part of the woodbury inverse 
        #w2 = np.linalg.solve(gp._Lnm, np.dot(gp._V,w1)) 
        #iGKnmy = np.dot(iGKnm.T, y-my)
        
        #l1 = np.dot(w1,w1) - np.dot(w2,w2)
        #l1 = np.dot(y, iGy) - np.dot(iGKnmy, np.dot(gp._iQ,iGKnmy))
        l1 = np.dot(y, iGy) - np.dot(r, alpha)
        #l2 = 2.0*detG + 2.0*np.sum(np.log(np.diag(gp._Lnm)))
        l2 = -2.0*np.sum(np.log(np.diag(gp._Lm)))+2.0*np.sum(np.log(np.diag(gp._Lq)))+2.0*detG
        
        likel = (-l1 - l2 - n*np.log(2.0*np.pi)) / 2.0
        return likel
    
    def gradient(self):
        return self.gradientFast()
      
    def gradientFast(self):
        '''
            Very Fast gradient computation of the kernel hyperparameters and inducing data points 
        by using chain rule of the likelihood derivative (see Alvarez 2011).
        
        '''
        gp = self._gp
        
        meanfct = gp._meanfct
        
        #data 
        X = gp._X
        y = gp._y
        Xu = gp._Xu
        
        n = len(X)
        m = len(Xu)
        
        my = np.zeros(len(y))
        if meanfct != None:
            my = meanfct(X)
        
        #precomuted stuff
        iKm = gp._iKm
        Knm = gp._Knm
        iG = gp._iG
        alpha = gp._alpha
        iQ = gp._iQ
        kernel = gp._kernel
        noise_kernel = gp._noise_kernel
        
        #formulas uses identical notation to the Paper of Alvarez
        #so the diag matrices are computed in a efficient way

        igroup = gp._igroup
        k = len(igroup)
        igroup = np.r_[igroup, n]

    
        t = time.time()
        
        
        #t1 = time.time()
        C = iQ + np.outer(alpha, alpha)
        B1 = np.dot(iKm, Knm.T)
        B3 = np.dot(Knm, alpha)
        
        
        #print 'grad1-timeA={0}'.format(time.time()-t1)
        #yy = np.zeros((n,n))
        H = np.empty(k, dtype=object)
        Q = np.empty(k, dtype=object)
        dKn = np.empty(k, dtype=object)
        #H = G+np.outer(y-my,y-my)+np.outer(np.dot(Knm, alpha), y-my)+np.outer(np.dot(Knm, alpha), y-my).T
        #J = H - np.dot(np.dot(Knm,C), Knm.T)
        #yy = np.outer(y-my, y-my)
        B2 = np.zeros((m,n))
        iGKnm = np.zeros((n,m))
        #t1 = time.time()
        for i in xrange(k):
            start = igroup[i]
            end = igroup[i+1]
            #t2 = time.time()
            #yy = np.outer(y[start:end]-my[start:end], y[start:end]-my[start:end])
            #b = cho_solve((Lg[start:end,start:end],1), y[start:end]-my[start:end])
            #B = np.outer(B3[start:end], y[start:end]-my[start:end])
            
            yy = np.outer(gp._iGy[start:end], gp._iGy[start:end])
            iGKnm[start:end] = np.dot(iG[i],Knm[start:end])
            #iGKnm = np.dot(iG[i], Knm)
            B = np.outer(np.dot(gp._iG[i],B3[start:end]), gp._iGy[start:end])
            #print 'grad2-timeA={0}'.format(time.time()-t2)
            
            
            #t2 = time.time()
            #H[i] = G[i] - yy + B + B.T
            #J[i] = H[i] - np.dot(np.dot(Knm[start:end],C), Knm[start:end].T)
            #Q[i] = np.dot(np.dot(iG[i], J[i]), iG[i])
            #print 'grad2-timeB={0}'.format(time.time()-t2)
            
            #t2 = time.time()
            H[i] = iG[i] - yy + B + B.T
            Q[i] = H[i] - np.dot(np.dot(iGKnm[start:end],C), iGKnm[start:end].T)
            #print 'grad2-timeB1={0}'.format(time.time()-t2)
            
            #print 'Qi={0}'.format(Q[i])
            #print 'Q1={0}'.format(Q1)
            
            #t2 = time.time()
            B2[:,start:end] = np.dot(B1[:,start:end], Q[i])
            #print 'grad2-timeC={0}'.format(time.time()-t2)
            
            #t2 = time.time()
            dKn[i] = -0.5*Q[i]
            #print 'grad2-timeD={0}'.format(time.time()-t2)
        
        #print 'grad1-timeB={0}'.format(time.time()-t1)
        
        #yy = (y-my)*(y-my)    
        #H = G - yy + 2.0*B3*(y-my)
        #J = H-np.sum(np.dot(Knm, C)*Knm,1)
        #Q = J/(G**2)
        #B2 = np.dot(B1,Q)
        
        t1 = time.time()
        dKm = 0.5*(iKm - C - np.dot(B2,B1.T))
        #dKm = 0.5*(iKm - C - np.dot(np.dot(np.dot(iKm, Knm.T),Q), np.dot(Knm, iKm)))
        dKmn = B2 - np.dot(C, iGKnm.T) + np.outer(alpha,gp._iGy)
        #dKmn = B2 - np.dot(C, np.dot(Knm.T,iG)) + np.outer(alpha,np.dot((y-my),iG))
        dKmn = dKmn.T
        #print 'grad precomp={0}'.format(time.time()-t)
        
        #print 'grad1-timeC={0}'.format(time.time()-t1)
#
#        t = time.time()
#        p = kernel.nparams
#        grad = np.empty(p)
#        t1 = time.time()
#        for i in xrange(p):
#            dkernel = kernel.derivate(i)
#            
#            dTm = dkernel(Xu, Xu)
#            dTnm = dkernel(X, Xu)
#            grad[i] = np.dot(dTm.flatten(), dKm.flatten()) 
#            grad[i] += np.dot(dTnm.flatten(), dKmn.flatten())
#            
#            #dTn = np.zeros((n,n))
#            for j in xrange(k):
#                start = igroup[j]
#                end = igroup[j+1]
#                
#                dTn = dkernel(X[start:end])
#                grad[i] += np.dot(dTn.flatten(), dKn[j].flatten())
#           
#        print 'grad1-timeD={0}'.format(time.time()-t1)
#        print 'gradkernel={0}'.format(grad)
        
        #t1 = time.time()
        grad = kernel.gradient(dKm, Xu, Xu)
        grad += kernel.gradient(dKmn, X, Xu)
        for i in xrange(k):
            start = igroup[i]
            end = igroup[i+1]
                
            #dTn = dkernel(X[start:end])
            grad += kernel.gradient(dKn[i], X[start:end])
        #print 'grad1-timeD={0}'.format(time.time()-t1)
        
        #t1 = time.time()
        if noise_kernel != None:
#            p = noise_kernel.nparams
#            gradNK = np.zeros(p)
#            for i in xrange(p):
#                dkernel = noise_kernel.derivate(i)
#                for j in xrange(k):
#                    start = igroup[j]
#                    end = igroup[j+1]
#            
#                    dTn = dkernel(X[start:end])
#                    gradNK[i] += np.dot(dTn.flatten(), dKn[j].flatten())
#            grad = np.r_[grad, gradNK]
               
            p = noise_kernel.nparams
            gradNK = np.zeros(p)
            for i in xrange(k):
                start = igroup[i]
                end = igroup[i+1]
                
                #dTn = dkernel(X[start:end])
                gradNK += noise_kernel.gradient(dKn[i], X[start:end])
            grad = np.r_[grad, gradNK]
        
        #print 'grad1-timeE={0}'.format(time.time()-t1)
        #print 'grad kernel params={0}'.format(time.time()-t)
        
        #determine the gradients of the mean fct hyperparameters if present
        if meanfct != None:
            p = meanfct.nparams
            gradM = np.empty(p)
            dM = np.zeros(n)
            for i in xrange(k):
                start = igroup[i]
                end = igroup[i+1]  
                dM[start:end] = np.dot((y-my),iG[i]) - np.dot(iG[i],B3)
            for i in xrange(k):
                d_meanfct = meanfct.derivate(i)
                dTM = d_meanfct(X)
                gradM[i] = np.dot(dTM, dM) #check if the gradient is correct
            grad = np.r_[grad, gradM]
            

        #t1 = time.time()
        if not gp._fix_inducing:
            #t1 = time.time()
            dkernelX = kernel.derivateX()
            dXm = dkernelX(Xu,Xu)*2.0
            dXmn = dkernelX(Xu,X)
            #print 'grad incducing params partI={0}'.format(time.time()-t1)
            m,d = Xu.shape
            gradX = np.zeros((m,d))
            
            #t1 = time.time()
            for i in xrange(m):
                for j in xrange(d):
                    gradX[i,j] = np.dot(dXm[:,i,j], dKm[:,i])
                    gradX[i,j] += np.dot(dXmn[:,i,j], dKmn[:,i])
            #print 'grad incducing params partII={0}'.format(time.time()-t1)
            grad = np.r_[grad, gradX.flatten()]
        #print 'grad incducing params={0}'.format(time.time()-t)
        #print 'grad1-timeF={0}'.format(time.time()-t1)

        return grad 


def _update_model(gp):
    X = gp._X           #training cases
    y = gp._y
         
    kernel = gp._kernel #kernel function to use 
    meanfct = gp._meanfct
    likel_fun = gp._likel_fun #log likelihood fun of the gp model
    
    #determine the cov matrix K and compute lower matrix L
    K = kernel(X)
    L = np.linalg.cholesky(K)
    
    #determin the mean vector if the mean function is specified
    m = np.zeros(len(y))
    if meanfct != None:
        m = meanfct(X)
    
    #compute alpha = K^(-1)*y 
    alpha = cho_solve((L,1), y-m)
    
    gp._K = K
    gp._L = L
    gp._alpha = alpha
    
    
    #compute the model log likelihood (gp relevant parameters must be updated before)
    #todo: check if it logical to update the likel here        
    likel = likel_fun()
    gp._log_likel = likel

    
if __name__ == '__main__':
    import scipy.optimize as spopt
    
    
    from upgeo.base.kernel import SEKernel, NoiseKernel
    from upgeo.base.gp import GPRegression
    
    
    P = np.asarray([1,1])
    
    kernel = SEKernel(np.log(2), np.log(21)) #+ NoiseKernel(np.log(0.5))
    #meanfct = BiasedLinearMean(np.array([0.3, -0.2, 0.8]), 2.1)
    meanfct = HiddenMean(LinearMean(np.array([0.3, -0.2, 0.8])) + ConstantMean([2.1]))
    #kernel = ARDSEKernel(np.log(1)*np.ones(3), np.log(5)) #+ NoiseKernel(np.log(0.5))
    
    X =  np.array( [[-0.5046,    0.3999,   -0.5607],
                    [-1.2706,   -0.9300,    2.1778],
                    [-0.3826,   -0.1768,    1.1385],
                    [0.6487,   -2.1321,   -2.4969],
                    [0.8257,    1.1454,    0.4413],
                    [-1.0149,   -0.6291,   -1.3981],
                    [-0.4711,   -1.2038,   -0.2551],
                    [0.1370,   -0.2539,    0.1644],
                    [-0.2919,   -1.4286,    0.7477],
                    [0.3018,   -0.0209,   -0.2730]])
    y = np.random.randn(10)
    
    gp = GPRegression(kernel, meanfct)
    gp.fit(X,y)
    likel_fun = gp.likel_fun
    
    
    def _l(p):
        gp.kernel.params = p[0:2]
        #gp.meanfct.params = p[2:6]
        _update_model(gp)
        return likel_fun()
    
    def _g(p):
        gp.kernel.params = p[0:2]
        #gp.meanfct.params = p[2:6]
        _update_model(gp)
        return likel_fun.gradient()

    print _l(np.log(np.array([0.2, 20])))
    print _g(np.log(np.array([0.2, 20])))
    #print _l(np.r_[np.log(np.array([0.2, 20])),np.array([-0.3, 0.5, 0.2, 2.3])])
    #print _g(np.r_[np.log(np.array([0.2, 20])),np.array([-0.3, 0.5, 0.2, 2.3])])
    print spopt.approx_fprime(np.log(np.array([0.2, 20])), _l, np.sqrt(np.finfo(float).eps))
    #print spopt.approx_fprime(np.r_[np.log(np.array([0.2, 20])),np.array([-0.3, 0.5, 0.2, 2.3])], _l, np.sqrt(np.finfo(float).eps))
    