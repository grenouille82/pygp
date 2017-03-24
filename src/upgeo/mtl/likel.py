'''
Created on Sep 25, 2012

@author: marcel
'''
import numpy as np

import time

from upgeo.base.likel import LikelihoodFunction
from scipy.linalg.decomp_cholesky import cho_solve
from upgeo.util.glob import APPROX_TYPE


#from upgeo.mtl.gp import APPROX_TYPE

class CMOGPGaussianLogLikel(LikelihoodFunction):
    '''
    @todo: - consider to define the input noise implicit in the likelihood function. 
             at the moment input noise can be formulated by the covariance function 
             as hyperparameter. 
    '''
    
    def __init__(self, gp_model):
        LikelihoodFunction.__init__(self, gp_model)
        
    def __call__(self):
        gp = self._gp
        
        y = gp._y           #targets of the training cases
        L = gp._L           #cholesky decomposition of the covariance matrix
        n = gp._n           #number of training samples
        alpha = gp._alpha   #weight vector for each kernel component???
        
        likel = (-np.dot((y).T, alpha) - n*np.log(2.0*np.pi)) / 2.0 #check, if we net the transpose
        likel -= np.sum(np.log(np.diag(L)))
                           
        #print 'likel={0}'.format(likel)
        
        return likel
    
    def gradient(self):
        gp = self._gp
        
        #@todo look for a better way to store the proximity
        
        X = gp._X           #covariates of the training cases
        alpha = gp._alpha   #weight vector for each kernel component???
        iK = gp._iK
        kernel = gp._kernel #used kernel in gp model
        itask = gp._itask
        
        
        #print 'grad'
        #K_inv = cho_solve((L,1), np.eye(n))
        Q = (np.outer(alpha, alpha)-iK)/2
        grad = kernel.gradient(Q, X, itaskX=itask)
        
        #print 'gradient={0}'.format(grad)
        return grad

class SparseCMOGPGaussianLogLikel(LikelihoodFunction):
    '''
    @todo: - consider to define the input noise implicit in the likelihood function. 
             at the moment input noise can be formulated by the covariance function 
             as hyperparameter. 
    '''
    
    def __init__(self, gp_model):
        LikelihoodFunction.__init__(self, gp_model)
        
    def __call__(self):
        gp = self._gp
        
        #print 'likel gp_hyperparams={0}'.format(gp.hyperparams)
        #print 'likel kernel_hyperparams={0}'.format(gp.kernel.params)
        
        y = gp._y           #targets of the training cases
        n = gp._n           #number of training samples
        Lu = gp._Lu 
        La = gp._La 
        Ld = gp._Ld
        iDy = gp._iDy
        
        k = gp._ntasks
        
        r = gp._r
        alpha = gp._alpha   #weight vector for each kernel component???
        
        approx = gp._approx_type
        
        if approx == APPROX_TYPE.FITC:
            detD = np.sum(np.log(Ld))
            #Q = np.diag(gp._D) + np.dot(np.dot(gp._Kfu, gp._iKu), gp._Kfu.T)
        elif approx == APPROX_TYPE.PITC:
            detD = 0
            #W = np.zeros((n,n))
            for i in xrange(k):
                #start = gp._itask[i]
                #end = gp._itask[i+1] if i != k-1 else n
                detD += np.sum(np.log(np.diag(Ld[i])))
                #print gp._iD[i].shape
                #print gp._D[i].shape
                #print W[start:end,start:end].shape
                #W[start:end,start:end] = gp._D[i]
            #Q = W + np.dot(np.dot(gp._Kfu, gp._iKu), gp._Kfu.T)
            
        else:
            raise TypeError('Unknown approx method')
        
        l1 = 2.0 * (detD - np.sum(np.log(np.diag(Lu))) + np.sum(np.log(np.diag(La))))
        #print 'l1={0}'.format(l1)
        #print 'l1a={0}'.format(np.linalg.slogdet(Q)[1])
        l2 = np.dot(y, iDy) - np.dot(r, alpha)
        #print 'l2={0}'.format(l2)
        #print 'l2a={0}'.format(np.dot(np.dot(y, np.linalg.inv(Q)), y))
        #print 'l2b={0}'.format(np.dot(y, iDy)-np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(y, np.diag(gp._iD)), gp._Kfu), gp._iA), gp._Kfu.T), np.diag(gp._iD)), y))
        #print 'l2b={0}'.format(np.dot(y, iDy)-np.dot(np.dot(np.dot(np.dot(y/gp._D, gp._Kfu), np.linalg.inv(gp._A)), gp._Kfu.T), y/gp._D))
        #print np.dot(np.dot(y, np.diag(gp._iD)), gp._Kfu) -np.dot(y/gp._D, gp._Kfu)
        #print np.linalg.inv(gp._A) - gp._iA
        #likel1 = (-np.dot(np.dot(y, np.linalg.inv(Q)), y) - n*np.log(2.0*np.pi)) / 2.0
        #likel1 -= np.log(np.linalg.det(Q))/2.0
        #likel1 -= np.linalg.slogdet(Q)[1]/2.0
        likel = (-l1 - l2 - n*np.log(2.0*np.pi)) / 2.0
        #print 'likel={0}'.format(likel)
        #print 'likel1={0}'.format(likel1)
        #print 'Q={0}'.format(Q)
        return likel
    
    def gradient(self):
        gp = self._gp
        
        #@todo look for a better way to store the proximity
        
        X = gp._X           #covariates of the training cases
        Xu = gp._Xu
        n = gp._n           #number of training samples
        m = len(Xu)
        
        kernel = gp._kernel #used kernel in gp model
        beta = gp._beta
        itask = gp._itask
        k = gp._ntasks
        
        approx = gp._approx_type
        
        
        Kfu = gp._Kfu
        iKu = gp._iKu
        D = gp._D
        iD = gp._iD
        iDy = gp._iDy
        
        y = gp._y
        iA = gp._iA
        alpha = gp._alpha
        
        #print 'grad'
        #K_inv = cho_solve((L,1), np.eye(n))
        #Q = (np.outer(alpha, alpha)-iK)/2
        #grad = kernel.gradient(Q, X, itaskX=itask)
        
        B1 = np.dot(iKu, Kfu.T)
        B3 = np.dot(Kfu, alpha)
        C = iA + np.outer(alpha, alpha)
        if approx == APPROX_TYPE.FITC:
            yy = y*y    
       
            H = D - yy + 2.0*B3*y
            J = H-np.sum(np.dot(Kfu, C)*Kfu,1)
            Q = J/(D**2)
        
            B1 = np.dot(iKu, Kfu.T)
            B2 = B1*Q

            dKf = -0.5*Q
            dKu = 0.5*(iKu - C - np.dot(B2,B1.T))
            dKfu = B2 - np.dot(C, Kfu.T)/D + np.outer(alpha,y/D)
            dKfu = dKfu.T
        elif approx == APPROX_TYPE.PITC:
            H = np.empty(k, dtype=object)
            Q = np.empty(k, dtype=object)
            dKf = np.empty(k, dtype=object)

            B2 = np.zeros((m,n))
            iDKfu = np.zeros((n,m))
            for i in xrange(k):
                start = itask[i]
                end = itask[i+1] if i != k-1 else n
            
                yy = np.outer(iDy[start:end], iDy[start:end])
                iDKfu[start:end] = np.dot(iD[i],Kfu[start:end])
            
                B = np.outer(np.dot(iD[i],B3[start:end]), iDy[start:end])
            
                H[i] = iD[i] - yy + B + B.T
                Q[i] = H[i] - np.dot(np.dot(iDKfu[start:end],C), iDKfu[start:end].T)
                B2[:,start:end] = np.dot(B1[:,start:end], Q[i])
           
                dKf[i] = -0.5*Q[i]
                
            dKu = 0.5*(iKu - C - np.dot(B2,B1.T))
        
            dKfu = B2 - np.dot(C, iDKfu.T) + np.outer(alpha, iDy)
            dKfu = dKfu.T
        else:
            raise TypeError('Unknown approx method')

        t = time.time()
        grad = kernel.gradient(dKu, Xu)
        #print 'grad1={0}'.format(kernel.gradient(dKu, Xu))
        
        grad += kernel.gradient(dKfu, X, Xu, itask)
        #print 'grad2={0}'.format(kernel.gradient(dKfu, X, Xu, itask))
        if approx == APPROX_TYPE.FITC:
            grad += kernel.gradient(dKf, X, itaskX=itask, diag=True)
            
        elif approx == APPROX_TYPE.PITC:
            for i in xrange(k):
                start = itask[i]
                end = itask[i+1] if i != k-1 else n
                #print 'start={0}'.format(start)
                #print 'end={0}'.format(end)
                #print 'gradblock'
                grad += kernel.gradient_block(dKf[i], X[start:end], i=i)
                #print 'grad3={0}'.format(kernel.gradient_block(dKf[i], X[start:end], i=i))
        #print 'hypergrad_time={0}'.format(time.time()-t)
        if not gp._fix_inducing:
            t = time.time()
            #todo: its a hack by using just the shared latent kernel to compute the gradient
            #      of Xu. make it implicit in the multikernel
            #t1 = time.time()
            
            
            gradX = kernel.gradientX(dKu, Xu, Xu) 
            #gradX = kernel.gradientX(dKu, Xu)
            #gradX += kernel.gradientX(dKfu, Xu, X, itaskZ=itask)
            gradX += kernel.gradientX(dKfu, X, Xu, itask)
            
            grad = np.r_[grad, gradX.flatten()]
            #print 'indgrad_time={0}'.format(time.time()-t)
        
        
        if beta != None:
        
            gradBeta = np.empty(k)
            for i in xrange(k): 
                gradBeta[i] = 0.5*beta[i]**(-2.0) * np.trace(Q[i])
            grad = np.r_[grad, gradBeta]
            #print 'gradBeta={0}'.format(gradBeta)
        #print 'gradient={0}'.format(grad)
        
        return grad

