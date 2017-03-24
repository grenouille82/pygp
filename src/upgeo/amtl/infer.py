'''
Created on May 2, 2012

@author: marcel
'''
import numpy as np
import scipy.optimize as spopt

from upgeo.base.infer import InferenceAlgorithm, ExactInference
from numpy.linalg.linalg import LinAlgError
from scipy.linalg.decomp_cholesky import cho_solve
from upgeo.base.gp import GPRegression
from upgeo.base.kernel import CompoundKernel, ExpARDSEKernel, ExpGaussianKernel,\
    ZeroKernel, HiddenKernel
from upgeo.util.math import fmin_ern
from upgeo.util.metric import mspe
from upgeo.amtl.util import gendata_fmtlgp_1d
from upgeo.util.type import enum

ApproxType = enum('FITC', 'PITC')

class FocusedOnePassInference(InferenceAlgorithm):
    
    __slots__ = ('_approx')
    
    def __init__(self, approx=ApproxType.PITC):
        '''
        '''
        self._approx = approx
    
    def _get_approx_type(self):
        return self._approx
    
    approx_type = property(fget=_get_approx_type)
    
    
    def apply(self, gp):
        approx = self._approx
        #training cases
        Xp = gp._Xp
        yp = gp._yp
        Xs = gp._Xs
        ys = gp._ys
        
        mp = gp._mp
        ms = gp._ms
        mtasks = gp._mtasks
        itask = np.r_[gp._itask, ms] #add the total number of sec tasks for easier iteration
        task_sizes = gp._task_sizes
       
        pKernel = gp._pKernel #kernel for the data of primary task
        sKernels = gp._sKernels #kernel for the data of the secondary tasks
        rhos = np.exp(gp._rhos) #rhos in log space->transform to linear space

        likel_fun = gp._likel_fun #log likelihood fun of the gp model
        
        Kp = pKernel(Xp)
        
        Lp = np.linalg.cholesky(Kp)
        Lpy = np.linalg.solve(Lp, yp)       #Lp^(-1)*y
        iKp = cho_solve((Lp, 1), np.eye(mp))
        Ksp = pKernel(Xs, Xp)
        Ksp = (Ksp.T*rhos.repeat(task_sizes)).T #multiply by the correlation parameter
        
        #Lq = np.linalg.solve(Lp, Ksp.T)     #chol(Lp)^-1 * Kps
        Lq = np.linalg.solve(np.linalg.cholesky(pKernel(Xp,Xp)+1e-6*np.eye(mp)), Ksp.T)     #chol(Lp)^-1 * Kps # noise-free alternative

        rp = np.dot(iKp, yp)                #iKp*yo
        rsp = np.dot(Ksp, rp)               #Ksp*iKp*yp
        
        #noise free stuff
        nfiKp = np.linalg.inv(pKernel(Xp,Xp)+1e-6*np.eye(mp))
        nfrp = np.dot(nfiKp, yp)
        #rsp = np.dot(Ksp, nfrp)

        Kpriv = np.zeros((ms,ms))
        D = np.zeros((ms,ms))               #D = G + Kpriv
        
        #compute the low rank approximation of G. (diagKs and G is either a diagonal matrix (FITC approx)
        #or a blkdiagonal matrix (PITC approx)
        #noise is include in the diagonal of Kp and Ks. So, maybe its better to use noise free
        #cov matrix for the approximation of the secondary tasks. (Contact the authors)
        #But its just an issue if the pKernel contains a noise term
        if approx == ApproxType.FITC:
            #G = diag(Ks - Ksp*iKp*Kps)
            #diagKs = pKernel(Xs, diag=True) * np.repeat(rhos**2.0, task_sizes) #maybe trouble with noise
            diagKs = np.diag(pKernel(Xs, Xs)) * np.repeat(rhos**2.0, task_sizes) #noise free alterniative
            G = diagKs - np.sum(Lq*Lq, 0)
            #compute block diagonal matrices
            for i in xrange(mtasks):
                start = itask[i]
                end = itask[i+1]
            
                Kxs = sKernels[i](Xs[start:end])    
                D[start:end, start:end] = Kxs + np.diag(G[start:end])
                Kpriv[start:end, start:end] = Kxs
                                
        elif approx == ApproxType.PITC:
            diagKs = np.zeros((ms,ms))
            G = np.zeros((ms,ms))
            Q = np.dot(Lq.T, Lq)
            for i in xrange(mtasks):
                start = itask[i]
                end = itask[i+1]
                
                #diagKs[start:end,start:end] = pKernel(Xs[start:end]) * rhos[i]**2
                diagKs[start:end,start:end] = pKernel(Xs[start:end],Xs[start:end]) * rhos[i]**2 #noise free alternative
                G[start:end,start:end] = diagKs[start:end,start:end]-Q[start:end,start:end]
                
                Kxs = sKernels[i](Xs[start:end])
                D[start:end, start:end] = Kxs + G[start:end,start:end]
                Kpriv[start:end, start:end] = Kxs
        else:
            raise ValueError('Unknown approx type: {0}'.format(approx))
                
        iD = np.zeros((ms,ms))              #inv(D)
        Ld = np.zeros((ms,ms))              #Ld = chol(D)
        rs = np.zeros(ms)                   #iD*ys
        rdsp = np.zeros(ms)                 #iD*Ksp*iKp*yp
        for i in xrange(mtasks):
            start = itask[i]
            end = itask[i+1]

            Ld[start:end, start:end] = np.linalg.cholesky(D[start:end, start:end])
            iD[start:end, start:end] = cho_solve((Ld[start:end,start:end], 1), np.eye(task_sizes[i]))
            
            rs[start:end] = np.dot(iD[start:end,start:end], ys[start:end])
            #rdsp[start:end] = np.dot(iD[start:end,start:end], np.dot(Ksp[start:end,:], rp))
            rdsp[start:end] = np.dot(iD[start:end,start:end], np.dot(Ksp[start:end,:], nfrp))#noise free

        rps = np.dot(Ksp.T, rs)             #Kps*iD*ys
        #rips = np.dot(iKp, rps)             #iKp*Kps*iD*ys
        rips = np.dot(nfiKp, rps)             #noise free
            
        alpha_p = rp + np.dot(nfiKp, np.dot(Ksp.T, rdsp)) - rips
        alpha_s = -rdsp + rs

        #print np.dot(Kpriv, Ksp)
        gp._Kp = Kp
        gp._Lp = Lp
        gp._Lpy = Lpy
        gp._iKp = iKp
        gp._Ksp = Ksp
        gp._diagKs = diagKs
        gp._Kpriv = Kpriv
        gp._Lq = Lq
        gp._G = G
        gp._D = D
        gp._iD = iD
        gp._Ld = Ld
        gp._rp = rp
        gp._rs = rs
        gp._rsp = rsp
        gp._rdsp = rdsp
        gp._rps = rps
        gp._rips = rips
        gp._alpha_p = alpha_p
        gp._alpha_s = alpha_s
        
        #compute the model log likelihood (gp relevant parameters must be updated before)
        #todo: check if it logical to update the likel here        
        likel = likel_fun()
        gp._log_likel = likel      
#
#        approx = self._approx
#        
#        #training cases
#        Xp = gp._Xp
#        yp = gp._yp
#        Xs = gp._Xs
#        ys = gp._ys
#        
#        mp = gp._mp
#        ms = gp._ms
#        mtasks = gp._mtasks
#        itask = np.r_[gp._itask, ms] #add the total number of sec tasks for easier iteration
#        task_sizes = gp._task_sizes
#       
#        pKernel = gp._pKernel #kernel for the data of primary task
#        sKernels = gp._sKernels #kernel for the data of the secondary tasks
#        rhos = np.exp(gp._rhos) #rhos in log space->transform to linear space
#
#        likel_fun = gp._likel_fun #log likelihood fun of the gp model
#        
#        Kp = pKernel(Xp)
#        
#        Lp = np.linalg.cholesky(Kp)
#        Lpy = np.linalg.solve(Lp, yp)       #Lp^(-1)*y
#        iKp = cho_solve((Lp, 1), np.eye(mp))
#        Ksp = pKernel(Xs, Xp)
#        Ksp = (Ksp.T*rhos.repeat(task_sizes)).T #multiply by the correlation parameter
#        
#        Lq = np.linalg.solve(Lp, Ksp.T)     #chol(Lp)^-1 * Kps
#
#        rp = np.dot(iKp, yp)                #iKp*yo
#        rsp = np.dot(Ksp, rp)               #Ksp*iKp*yp
#
#        Kpriv = np.zeros((ms,ms))
#        D = np.zeros((ms,ms))               #D = G + Kpriv
#        
#        #compute the low rank approximation of G. (diagKs and G is either a diagonal matrix (FITC approx)
#        #or a blkdiagonal matrix (PITC approx)
#        #noise is include in the diagonal of Kp and Ks. So, maybe its better to use noise free
#        #cov matrix for the approximation of the secondary tasks. (Contact the authors)
#        #But its just an issue if the pKernel contains a noise term
#        if approx == ApproxType.FITC:
#            #G = diag(Ks - Ksp*iKp*Kps)
#            diagKs = pKernel(Xs, diag=True) * np.repeat(rhos**2.0, task_sizes) #maybe trouble with noise
#            #diagKs = np.diag(pKernel(Xs, Xs)) * np.repeat(rhos**2.0, task_sizes) #noise free alterniative
#            G = diagKs - np.sum(Lq*Lq, 0)
#            #compute block diagonal matrices
#            for i in xrange(mtasks):
#                start = itask[i]
#                end = itask[i+1]
#            
#                Kxs = sKernels[i](Xs[start:end])    
#                D[start:end, start:end] = Kxs + np.diag(G[start:end])
#                Kpriv[start:end, start:end] = Kxs
#                                
#        elif approx == ApproxType.PITC:
#            diagKs = np.zeros((ms,ms))
#            G = np.zeros((ms,ms))
#            Q = np.dot(Lq.T, Lq)
#            for i in xrange(mtasks):
#                start = itask[i]
#                end = itask[i+1]
#                
#                diagKs[start:end,start:end] = pKernel(Xs[start:end]) * rhos[i]**2
#                G[start:end,start:end] = diagKs[start:end,start:end]-Q[start:end,start:end]
#                
#                Kxs = sKernels[i](Xs[start:end])
#                D[start:end, start:end] = Kxs + G[start:end,start:end]
#                Kpriv[start:end, start:end] = Kxs
#        else:
#            raise ValueError('Unknown approx type: {0}'.format(approx))
#                
#        iD = np.zeros((ms,ms))              #inv(D)
#        Ld = np.zeros((ms,ms))              #Ld = chol(D)
#        rs = np.zeros(ms)                   #iD*ys
#        rdsp = np.zeros(ms)                 #iD*Ksp*iKp*yp
#        for i in xrange(mtasks):
#            start = itask[i]
#            end = itask[i+1]
#
#            Ld[start:end, start:end] = np.linalg.cholesky(D[start:end, start:end])
#            iD[start:end, start:end] = cho_solve((Ld[start:end,start:end], 1), np.eye(task_sizes[i]))
#            
#            rs[start:end] = np.dot(iD[start:end,start:end], ys[start:end])
#            rdsp[start:end] = np.dot(iD[start:end,start:end], np.dot(Ksp[start:end,:], rp))
#
#        
#        rps = np.dot(Ksp.T, rs)             #Kps*iD*ys
#        rips = np.dot(iKp, rps)             #iKp*Kps*iD*ys
#            
#        alpha_p = rp + np.dot(iKp, np.dot(Ksp.T, rdsp)) - rips
#        alpha_s = -rdsp + rs
#
#        #print np.dot(Kpriv, Ksp)
#        gp._Kp = Kp
#        gp._Lp = Lp
#        gp._Lpy = Lpy
#        gp._iKp = iKp
#        gp._Ksp = Ksp
#        gp._diagKs = diagKs
#        gp._Kpriv = Kpriv
#        gp._Lq = Lq
#        gp._G = G
#        gp._D = D
#        gp._iD = iD
#        gp._Ld = Ld
#        gp._rp = rp
#        gp._rs = rs
#        gp._rsp = rsp
#        gp._rdsp = rdsp
#        gp._rps = rps
#        gp._rips = rips
#        gp._alpha_p = alpha_p
#        gp._alpha_s = alpha_s
#        
#        #compute the model log likelihood (gp relevant parameters must be updated before)
#        #todo: check if it logical to update the likel here        
#        likel = likel_fun()
#        gp._log_likel = likel      
        
class FocusedExactInference(InferenceAlgorithm):

    __slots__ = ('_approx')
    
    def __init__(self, approx=ApproxType.PITC):
        '''
        '''
        self._approx = approx
    
    def _get_approx_type(self):
        return self._approx
    
    approx_type = property(fget=_get_approx_type)

    
    def apply(self, gp):
        '''
        '''
        approx = self._approx
        FocusedExactInference._update_model(gp, approx)
        
        params = gp.hyperparams
        print params
        
        f = FocusedExactInference._TargetFunction(gp, approx) #reference to fun to be optimized
        fprime = f.gradient                     #reference to the gradient
        cb = f.callback                         #reference to the callback fun
        x0 = np.copy(params)                    #initial guess of the hyperparameters
       
        #opt_result = spopt.fmin_bfgs(f, x0, fprime, callback=cb, maxiter=20, gtol=1e-2, full_output = True)
        #opt_result = spopt.fmin_bfgs(f, x0, None, callback=cb, maxiter=100, gtol=1e-2, full_output = True)
        opt_result = spopt.fmin_bfgs(f, x0, fprime, callback=cb, full_output = True)
        #opt_result = spopt.fmin_cg(f, x0, fprime, gtol=1e-2, callback=cb, full_output = True)
        #xopt = fmin_ern(f, x0, fprime, maxiter=500)
        xopt = opt_result[0]    #optimal hyperparameters with respect to log likelihood
        #110
        #Debug output
        if False:
            fun_calls = opt_result[4]
            grad_calls = opt_result[5]
            print 'number of opt calls: fun={0}, grad={1}'.format(fun_calls, grad_calls)
            
            err = spopt.check_grad(f, fprime, x0)
            print 'gradient check at {0}: error={1}'.format(x0, err)
            print spopt.approx_fprime(x0, f, np.sqrt(np.finfo(float).eps))
            
        
        FocusedExactInference._update_parameters(gp, xopt)
        FocusedExactInference._update_model(gp, approx)
            
    class _TargetFunction(object):
        
        __slots__ = ('_gp',
                     '_likel_fun',
                     '_approx'
                     )
        
        def __init__(self, gp, approx):
            self._gp = gp
            self._approx = approx
            self._likel_fun = gp._likel_fun
        
        def __call__(self, params):
            
            print 'parmas={0}'.format(np.exp(params))
            flag = np.logical_or(np.isnan(params), np.isinf(params))
            if np.any(flag):
                print 'bad params={0}'.format(params)
                return 1e+300

            
            gp = self._gp
            old_params = gp.hyperparams #todo 
            
            #FocusedExactInference._update_parameters(gp, params)
            #FocusedExactInference._update_model(gp, self._approx)
            try: 
                FocusedExactInference._update_parameters(gp, params)
                FocusedExactInference._update_model(gp, self._approx)
            except LinAlgError:
                FocusedExactInference._update_parameters(gp, old_params)
                print 'fucker'
                return 1e+300
            
            print 'gp likel={0}'.format(-gp._log_likel)
            #print 'gp likel params ={0}'.format(np.exp(params))
            #print 'likel={0}'.format(-gp._log_likel)
            return -gp._log_likel
        
        def gradient(self, params):
            #print 'params1={0}'.format(params)
            likel_fun = self._likel_fun
            print 'gradient={0}'.format(-likel_fun.gradient()) 
            grad = -likel_fun.gradient()
            #grad[-1] = 0
            flag = np.logical_or(np.isnan(grad), np.isinf(grad))
            if np.any(flag):
                grad[flag] = 0
                print 'bad gradient={0}'.format(grad)
            return grad
                
        def callback(self, params):
            gp = self._gp
            
            FocusedExactInference._update_parameters(gp, params)
            FocusedExactInference._update_model(gp, self._approx)
        
    
    @staticmethod
    def _update_parameters(gp, params):
        pKernel = gp._pKernel
        sKernels = gp._sKernels
        mtp = gp._mtp
        mts = gp._mts
        mtasks = gp._mtasks
        
        theta_p = params[:mtp]
        theta_s = params[mtp:mtp+mtasks*mts]
        theta_s = np.reshape(theta_s, (mtasks, mts))
        rhos = params[mtp+mtasks*mts:]
        
        pKernel.params = theta_p
        for i in xrange(mtasks):
            sKernels[i].params = theta_s[i]
        
        #print 'update_rhos={0}'.format(rhos)
        gp._rhos = rhos
        
    @staticmethod
    def _update_model(gp, approx):
    
        #training cases
        Xp = gp._Xp
        yp = gp._yp
        Xs = gp._Xs
        ys = gp._ys
        
        mp = gp._mp
        ms = gp._ms
        mtasks = gp._mtasks
        itask = np.r_[gp._itask, ms] #add the total number of sec tasks for easier iteration
        task_sizes = gp._task_sizes
       
        pKernel = gp._pKernel #kernel for the data of primary task
        sKernels = gp._sKernels #kernel for the data of the secondary tasks
        rhos = np.exp(gp._rhos) #rhos in log space->transform to linear space

        likel_fun = gp._likel_fun #log likelihood fun of the gp model
        
        Kp = pKernel(Xp)
        
        Lp = np.linalg.cholesky(Kp)
        Lpy = np.linalg.solve(Lp, yp)       #Lp^(-1)*y
        iKp = cho_solve((Lp, 1), np.eye(mp))
        Ksp = pKernel(Xs, Xp)
        Ksp = (Ksp.T*rhos.repeat(task_sizes)).T #multiply by the correlation parameter
        
        #Lq = np.linalg.solve(Lp, Ksp.T)     #chol(Lp)^-1 * Kps
        Lq = np.linalg.solve(np.linalg.cholesky(pKernel(Xp,Xp)+1e-6*np.eye(mp)), Ksp.T)     #chol(Lp)^-1 * Kps # noise-free alternative

        rp = np.dot(iKp, yp)                #iKp*yo
        rsp = np.dot(Ksp, rp)               #Ksp*iKp*yp
        
        #noise free stuff
        nfiKp = np.linalg.inv(pKernel(Xp,Xp)+1e-6*np.eye(mp))
        nfrp = np.dot(nfiKp, yp)
        #rsp = np.dot(Ksp, nfrp)

        Kpriv = np.zeros((ms,ms))
        D = np.zeros((ms,ms))               #D = G + Kpriv
        
        #compute the low rank approximation of G. (diagKs and G is either a diagonal matrix (FITC approx)
        #or a blkdiagonal matrix (PITC approx)
        #noise is include in the diagonal of Kp and Ks. So, maybe its better to use noise free
        #cov matrix for the approximation of the secondary tasks. (Contact the authors)
        #But its just an issue if the pKernel contains a noise term
        if approx == ApproxType.FITC:
            #G = diag(Ks - Ksp*iKp*Kps)
            #diagKs = pKernel(Xs, diag=True) * np.repeat(rhos**2.0, task_sizes) #maybe trouble with noise
            diagKs = np.diag(pKernel(Xs, Xs)) * np.repeat(rhos**2.0, task_sizes) #noise free alterniative
            G = diagKs - np.sum(Lq*Lq, 0)
            #compute block diagonal matrices
            for i in xrange(mtasks):
                start = itask[i]
                end = itask[i+1]
            
                Kxs = sKernels[i](Xs[start:end])    
                D[start:end, start:end] = Kxs + np.diag(G[start:end])
                Kpriv[start:end, start:end] = Kxs
                                
        elif approx == ApproxType.PITC:
            diagKs = np.zeros((ms,ms))
            G = np.zeros((ms,ms))
            Q = np.dot(Lq.T, Lq)
            for i in xrange(mtasks):
                start = itask[i]
                end = itask[i+1]
                
                #diagKs[start:end,start:end] = pKernel(Xs[start:end]) * rhos[i]**2
                diagKs[start:end,start:end] = pKernel(Xs[start:end],Xs[start:end]) * rhos[i]**2 #noise free alternative
                G[start:end,start:end] = diagKs[start:end,start:end]-Q[start:end,start:end]
                
                Kxs = sKernels[i](Xs[start:end])
                D[start:end, start:end] = Kxs + G[start:end,start:end]
                Kpriv[start:end, start:end] = Kxs
        else:
            raise ValueError('Unknown approx type: {0}'.format(approx))
                
        iD = np.zeros((ms,ms))              #inv(D)
        Ld = np.zeros((ms,ms))              #Ld = chol(D)
        rs = np.zeros(ms)                   #iD*ys
        rdsp = np.zeros(ms)                 #iD*Ksp*iKp*yp
        for i in xrange(mtasks):
            start = itask[i]
            end = itask[i+1]

            Ld[start:end, start:end] = np.linalg.cholesky(D[start:end, start:end])
            iD[start:end, start:end] = cho_solve((Ld[start:end,start:end], 1), np.eye(task_sizes[i]))
            
            rs[start:end] = np.dot(iD[start:end,start:end], ys[start:end])
            #rdsp[start:end] = np.dot(iD[start:end,start:end], np.dot(Ksp[start:end,:], rp))
            rdsp[start:end] = np.dot(iD[start:end,start:end], np.dot(Ksp[start:end,:], nfrp))#noise free

        rps = np.dot(Ksp.T, rs)             #Kps*iD*ys
        #rips = np.dot(iKp, rps)             #iKp*Kps*iD*ys
        rips = np.dot(nfiKp, rps)             #noise free
            
        alpha_p = rp + np.dot(nfiKp, np.dot(Ksp.T, rdsp)) - rips
        alpha_s = -rdsp + rs

        #print np.dot(Kpriv, Ksp)
        gp._Kp = Kp
        gp._Lp = Lp
        gp._Lpy = Lpy
        gp._iKp = iKp
        gp._Ksp = Ksp
        gp._diagKs = diagKs
        gp._Kpriv = Kpriv
        gp._Lq = Lq
        gp._G = G
        gp._D = D
        gp._iD = iD
        gp._Ld = Ld
        gp._rp = rp
        gp._rs = rs
        gp._rsp = rsp
        gp._rdsp = rdsp
        gp._rps = rps
        gp._rips = rips
        gp._alpha_p = alpha_p
        gp._alpha_s = alpha_s
        
        #compute the model log likelihood (gp relevant parameters must be updated before)
        #todo: check if it logical to update the likel here        
        likel = likel_fun()
        gp._log_likel = likel      
                        
class FMOGPOnePassInference(InferenceAlgorithm):
    
    __slots__ = ()
    
    def apply(self, gp):
        #training cases
        Xp = gp._Xp
        yp = gp._yp
        Xs = gp._Xs
        ys = gp._ys
        
        mp = gp._mp
        ms = gp._ms
        mtasks = gp._mtasks
        itask = np.r_[gp._itask, ms] #add the total number of sec tasks for easier iteration
        task_sizes = gp._task_sizes
        
        
        kernel = gp._kernel     #kernel for the data of primary task

        likel_fun = gp._likel_fun #log likelihood fun of the gp model
        
        Kp = kernel.cov(Xp, task=0, latent=True) + kernel.cov(Xp, task=0, latent=False)
        Lp = np.linalg.cholesky(Kp)
        Lpy = np.linalg.solve(Lp, yp)       #Lp^(-1)*y
        iKp = cho_solve((Lp, 1), np.eye(mp))
        
        Ksp = np.zeros((ms,mp))
        diagKs = np.zeros(ms)
        for i in xrange(mtasks):
            start = itask[i]
            end = itask[i+1]
            Ksp[start:end,:] = kernel.cross_cov(Xs[start:end], Xp, task=i+1, latent=True)
            diagKs[start:end] = kernel.cov(Xs[start:end], task=i+1, latent=True, diag=True)
        
        Lq = np.linalg.solve(Lp, Ksp.T)
        G = diagKs - np.sum(Lq*Lq, 0) #G = diag(Ks - Ksp*Kp^(-1)*Ksp.T)
        
        rp = np.dot(iKp, yp)                #iKp*yo
        rsp = np.dot(Ksp, rp)               #Ksp*iKp*yp
        
        #compute block diagonal matrices
        Kpriv = np.zeros((ms,ms))
        D = np.zeros((ms,ms))               #D = G + Kpriv
        iD = np.zeros((ms,ms))              #inv(D)
        Ld = np.zeros((ms,ms))              #Ld = chol(D)
        rs = np.zeros(ms)                   #iD*ys
        rdsp = np.zeros(ms)                 #iD*Ksp*iKp*yp
        for i in xrange(mtasks):
            start = itask[i]
            end = itask[i+1]
            
            Kxs = kernel.cov(Xs[start:end], task=i+1, latent=False)
            
            D[start:end, start:end] = Kxs + np.diag(G[start:end])
            Ld[start:end, start:end] = np.linalg.cholesky(D[start:end, start:end])
            iD[start:end, start:end] = cho_solve((Ld[start:end,start:end], 1), np.eye(task_sizes[i]))
            Kpriv[start:end, start:end] = Kxs
            
            rs[start:end] = np.dot(iD[start:end,start:end], ys[start:end])
            rdsp[start:end] = np.dot(iD[start:end,start:end], np.dot(Ksp[start:end,:], rp))
        
        rps = np.dot(Ksp.T, rs)             #Kps*iD*ys
        rips = np.dot(iKp, rps)             #iKp*Kps*iD*ys
            
        alpha_p = rp + np.dot(iKp, np.dot(Ksp.T, rdsp)) - rips
        alpha_s = -rdsp + rs

        #print np.dot(Kpriv, Ksp)
        gp._Kp = Kp
        gp._Lp = Lp
        gp._Lpy = Lpy
        gp._iKp = iKp
        gp._Ksp = Ksp
        gp._diagKs = diagKs
        gp._Kpriv = Kpriv
        gp._Lq = Lq
        gp._G = G
        gp._D = D
        gp._iD = iD
        gp._Ld = Ld
        gp._rp = rp
        gp._rs = rs
        gp._rsp = rsp
        gp._rdsp = rdsp
        gp._rps = rps
        gp._rips = rips
        gp._alpha_p = alpha_p
        gp._alpha_s = alpha_s
        
        #compute the model log likelihood (gp relevant parameters must be updated before)
        #todo: check if it logical to update the likel here        
        likel = likel_fun()
        gp._log_likel = likel
        
class FMOGPExactInference(InferenceAlgorithm):
    
    __slots__ = ()
    
    def apply(self, gp):
        '''
        '''
        FMOGPExactInference._update_model(gp)
        
        kernel = gp._kernel
        params = kernel.params
        
        f = FMOGPExactInference._TargetFunction(gp) #reference to fun to be optimized
       
        fprime = f.gradient                     #reference to the gradient
        cb = f.callback                         #reference to the callback fun
        x0 = np.copy(params)                    #initial guess of the hyperparameters
       
        opt_result = spopt.fmin_bfgs(f, x0, None, callback=cb, maxiter=100, gtol=1e-2, full_output = True)
        #opt_result = spopt.fmin_bfgs(f, x0, fprime, callback=cb, gtol=1e-2, full_output = True)
        #opt_result = spopt.fmin_bfgs(f, x0, fprime, callback=cb, full_output = True)
        #opt_result = spopt.fmin_cg(f, x0, fprime, gtol=1e-2, callback=cb, full_output = True)
        #xopt = fmin_ern(f, x0, fprime, maxiter=500)
        xopt = opt_result[0]    #optimal hyperparameters with respect to log likelihood
        
        #Debug output
        if False:
            fun_calls = opt_result[4]
            grad_calls = opt_result[5]
            print 'number of opt calls: fun={0}, grad={1}'.format(fun_calls, grad_calls)
            
            err = spopt.check_grad(f, fprime, x0)
            print 'gradient check at {0}: error={1}'.format(x0, err)
            print spopt.approx_fprime(x0, f, np.sqrt(np.finfo(float).eps))
            
        
        FMOGPExactInference._update_parameters(gp, xopt)
        FMOGPExactInference._update_model(gp)
            
    class _TargetFunction(object):
        
        __slots__ = ('_gp',
                     '_likel_fun'
                     )
        
        def __init__(self, gp):
            self._gp = gp
            self._likel_fun = gp._likel_fun
        
        def __call__(self, params):
            #print params
            flag = np.logical_or(np.isnan(params), np.isinf(params))
            if np.any(flag):
                print 'bad params={0}'.format(params)
                return 1e+300

            
            gp = self._gp
            #old_params = gp.hyperparams #todo 
            
            try: 
                FMOGPExactInference._update_parameters(gp, params)
                FMOGPExactInference._update_model(gp)
            except LinAlgError:
                #FocusedExactInference._update_parameters(gp, old_params)
                return 1e+300
            
            print 'gp likel={0}'.format(-gp._log_likel)
            #print 'gp likel params ={0}'.format(np.exp(params))
            
            return -gp._log_likel
        
        def gradient(self, params):
            #print 'params1={0}'.format(params)
            likel_fun = self._likel_fun
            #print 'gradient={0}'.format(-likel_fun.gradient()) 
            grad = -likel_fun.gradient()
            flag = np.logical_or(np.isnan(grad), np.isinf(grad))
            if np.any(flag):
                grad[flag] = 0
                print 'bad gradient={0}'.format(grad)
            return grad
                
        def callback(self, params):
            gp = self._gp
            
            FMOGPExactInference._update_parameters(gp, params)
            FMOGPExactInference._update_model(gp)
        
    
    @staticmethod
    def _update_parameters(gp, params):
        kernel = gp._kernel
        kernel.params = params
        
    @staticmethod
    def _update_model(gp):

        Xp = gp._Xp
        yp = gp._yp
        Xs = gp._Xs
        ys = gp._ys
        
        mp = gp._mp
        ms = gp._ms
        mtasks = gp._mtasks
        itask = np.r_[gp._itask, ms] #add the total number of sec tasks for easier iteration
        task_sizes = gp._task_sizes
        
        
        kernel = gp._kernel     #kernel for the data of primary task
        likel_fun = gp._likel_fun #log likelihood fun of the gp model
        
        Kp = kernel.cov(Xp, task=0, latent=True) + kernel.cov(Xp, task=0, latent=False)
        Lp = np.linalg.cholesky(Kp)
        Lpy = np.linalg.solve(Lp, yp)       #Lp^(-1)*y
        iKp = cho_solve((Lp, 1), np.eye(mp))
        
        Ksp = np.zeros((ms,mp))
        diagKs = np.zeros(ms)
        for i in xrange(mtasks):
            start = itask[i]
            end = itask[i+1]
            Ksp[start:end,:] = kernel.cross_cov(Xs[start:end], Xp, task=i+1, latent=True)
            diagKs[start:end] = kernel.cov(Xs[start:end], task=i+1, latent=True, diag=True)
        
        
        #print 'diagKS'
        #print diagKs
        Lq = np.linalg.solve(Lp, Ksp.T)
        G = diagKs - np.sum(Lq*Lq, 0) #G = diag(Ks - Ksp*Kp^(-1)*Ksp.T)
        
        rp = np.dot(iKp, yp)                #iKp*yo
        rsp = np.dot(Ksp, rp)               #Ksp*iKp*yp
        
        #compute block diagonal matrices
        Kpriv = np.zeros((ms,ms))
        D = np.zeros((ms,ms))               #D = G + Kpriv
        iD = np.zeros((ms,ms))              #inv(D)
        Ld = np.zeros((ms,ms))              #Ld = chol(D)
        rs = np.zeros(ms)                   #iD*ys
        rdsp = np.zeros(ms)                 #iD*Ksp*iKp*yp
        for i in xrange(mtasks):
            start = itask[i]
            end = itask[i+1]
            
            Kxs = kernel.cov(Xs[start:end], task=i+1, latent=False)
            
            D[start:end, start:end] = Kxs + np.diag(G[start:end])
            Ld[start:end, start:end] = np.linalg.cholesky(D[start:end, start:end])
            iD[start:end, start:end] = cho_solve((Ld[start:end,start:end], 1), np.eye(task_sizes[i]))
            Kpriv[start:end, start:end] = Kxs
            
            rs[start:end] = np.dot(iD[start:end,start:end], ys[start:end])
            rdsp[start:end] = np.dot(iD[start:end,start:end], np.dot(Ksp[start:end,:], rp))
        
        rps = np.dot(Ksp.T, rs)             #Kps*iD*ys
        rips = np.dot(iKp, rps)             #iKp*Kps*iD*ys
            
        alpha_p = rp + np.dot(iKp, np.dot(Ksp.T, rdsp)) - rips
        alpha_s = -rdsp + rs

        #print np.dot(Kpriv, Ksp)
        gp._Kp = Kp
        gp._Lp = Lp
        gp._Lpy = Lpy
        gp._iKp = iKp
        gp._Ksp = Ksp
        gp._diagKs = diagKs
        gp._Kpriv = Kpriv
        gp._Lq = Lq
        gp._G = G
        gp._D = D
        gp._iD = iD
        gp._Ld = Ld
        gp._rp = rp
        gp._rs = rs
        gp._rsp = rsp
        gp._rdsp = rdsp
        gp._rps = rps
        gp._rips = rips
        gp._alpha_p = alpha_p
        gp._alpha_s = alpha_s
        
        #compute the model log likelihood (gp relevant parameters must be updated before)
        #todo: check if it logical to update the likel here        
        
        likel = likel_fun()
        gp._log_likel = likel
        
if __name__ == '__main__':
    from upgeo.base.kernel import SEKernel, NoiseKernel
    from upgeo.mt.util import gendata_fmtlgp_2d
    
    from upgeo.mt.gp import FocusedGPRegression
    from upgeo.mt.likel import FocusedGaussianLogLikel
    
    
    from upgeo.base.kernel import DiracConvolvedKernel
    from upgeo.mt.kernel import DiracConvolvedAMTLKernel
    from upgeo.mt.gp import FMOGPRegression
    
    from upgeo.mt.likel import SimpleFocusedGaussianLogLikel, PITCFocusedGaussianLogLikel,FTCFocusedGaussianLogLikel
    
    from upgeo.mt.gp import AMOGPRegression
    from upgeo.mt.likel import PFITCFocusedGaussianLogLikel

    import matplotlib.pyplot as plt
    
    pKernel = SEKernel(np.log(0.5), np.log(1)) + NoiseKernel(np.log(0.2))
    sKernel = SEKernel(np.log(0.5), np.log(1)) + NoiseKernel(np.log(0.2))
    #sKernel = NoiseKernel(np.log(0.2))
    #pKernel = HiddenKernel(SEKernel(np.log(0.65028441), np.log(1.78366781)) + NoiseKernel(np.log(0.08415811)))
    #sKernel = HiddenKernel(SEKernel(np.log(9.65430200e-01), np.log(7.43907600e-05)) + NoiseKernel(np.log(6.83140308e-02)))
    rho = np.log(0.1)
    
    kernel = SEKernel(np.log(0.8), np.log(2))
    ntasks = 10
    #x, yp, ys = gendata_fmtlgp_1d(kernel, kernel, -5, 5, 100, ntasks, 0.3,343885482375)
    #x, yp, ys = gendata_fmtlgp_1d(kernel, kernel, -5, 5, 100, ntasks, 0.3,82387728)
    #x, yp, ys = gendata_fmtlgp_1d(kernel, kernel, -5, 5, 50, ntasks, 0.1, 2812336741234)
    x, yp, ys = gendata_fmtlgp_1d(kernel, kernel, -5, 5, 50, ntasks, 0.3)
    #x, yp, ys = gendata_fmtlgp_1d(kernel, kernel, -5, 5, 100, ntasks, 0.3, 43764387652)
    #x, yp, ys = gendata_fmtlgp_1d(kernel, kernel, -5, 5, 100, ntasks, 0.3, 437643)
    
    #plt.show()
    for i in xrange(ntasks):
        print 'task:{0}'.format(i)
        plt.plot(x,yp,'*')
        plt.plot(x,ys[:,i], 'r*')
        plt.show()
    #preprocess data
    Xp = x
    yp = yp
    #Xt = Xp[np.arange(0,100,7)]#np.r_[x[0:30], x[70:]]
    #yt = yp[np.arange(0,100,7)]#np.r_[yp[0:30], yp[70:]]
    Xt = np.r_[x[0:10], x[40:]]
    yt = np.r_[yp[0:10], yp[40:]]
    Xs = np.tile(Xp, (ntasks,1))
    ys = np.r_[ys.T.ravel()]
    #ys = np.r_[ys.T.ravel(), yp]
    itask =  np.arange(0,500,50)
    
    print Xp
    print yp
    print Xs
    print ys

    plt.plot(x,yp,'*')
    plt.plot(Xt,yt,'r*')
    plt.show()
    
#    gp = FocusedGPRegression(pKernel, sKernel, rho, likel_fun=PITCFocusedGaussianLogLikel, infer_method=FocusedExactInference(ApproxType.PITC))
#    gp.fit(Xt, yt, Xs, ys, itask)
#    print 'likel'
#    print gp._log_likel

#    gp = FocusedGPRegression(pKernel, sKernel, rho, likel_fun=FTCFocusedGaussianLogLikel, infer_method=FocusedOnePassInference(ApproxType.PITC))
#    gp.fit(Xt, yt, Xs, ys, itask)
#    print 'likel'
#    print gp._log_likel
#    yfit = gp.predict(Xp)
#    print 'mspe={0}'.format(mspe(yp, yfit))
#    print 'sjhwqoj'
#    print np.exp(gp.hyperparams)
#    plt.plot(Xp,yp,'*')
#    plt.plot(Xp, gp.predict(x))
#    plt.show()
#    
#    target_fct = FocusedExactInference._TargetFunction(gp, ApproxType.PITC)
#    params = np.copy(gp.hyperparams)
#    params = np.log(np.abs(np.random.randn(len(params))))
#    print 'likel={0}'.format(target_fct(params))
#    print 'gradient={0}'.format(target_fct.gradient(params))
#    print 'approx_gradient={0}'.format(spopt.approx_fprime(params, target_fct, np.sqrt(np.finfo(float).eps)))
#    print np.sqrt(np.finfo(float).eps)



    
#    gp = FocusedGPRegression(pKernel, sKernel, rho, likel_fun=FocusedGaussianLogLikel, infer_method=FocusedOnePassInference(ApproxType.FITC))
#    gp.fit(Xt, yt, Xs, ys, itask)
#    print 'likel'
#    print gp._log_likel

    #target_fct = FocusedExactInference._TargetFunction(gp, ApproxType.FITC)
    #params = np.copy(gp.hyperparams)
    #params = np.log(np.abs(np.random.randn(len(params))))
#    params[-5:] = np.log(0.5)
    #print 'likel={0}'.format(target_fct(params))
    #print 'gradient={0}'.format(target_fct.gradient(params))
    #print 'approx_gradient={0}'.format(spopt.approx_fprime(params, target_fct, np.sqrt(np.finfo(float).eps)))
    #print np.sqrt(np.finfo(float).eps)

    
    
#    gp = FocusedGPRegression(pKernel, sKernel, rho, likel_fun=FocusedGaussianLogLikel, infer_method=FocusedOnePassInference(ApproxType.PITC))
#    gp.fit(Xt, yt, Xs, ys, itask)
#    print 'likel'
#    print gp._log_likel


    #target_fct = FocusedExactInference._TargetFunction(gp, ApproxType.PITC)
    #params = np.copy(gp.hyperparams)
    #params = np.log(np.abs(np.random.randn(len(params))))
#    params[-5:] = np.log(0.5)
    #print 'likel={0}'.format(target_fct(params))
    #print 'gradient={0}'.format(target_fct.gradient(params))
    #print 'approx_gradient={0}'.format(spopt.approx_fprime(params, target_fct, np.sqrt(np.finfo(float).eps)))
    #print np.sqrt(np.finfo(float).eps)


    gp = AMOGPRegression(pKernel, sKernel, rho, likel_fun=PFITCFocusedGaussianLogLikel, infer_method=FocusedOnePassInference(ApproxType.FITC))
    gp.fit(Xt, yt, Xs, ys, itask)
    print 'likel'
    print gp._log_likel
    yfit = gp.predict(Xp)
    print 'mspe={0}'.format(mspe(yp, yfit))
    print 'sjhwqoj'
    print np.exp(gp.hyperparams)
    plt.plot(Xp,yp,'*')
    plt.plot(Xp, gp.predict(x))
    plt.show()
    
    target_fct = FocusedExactInference._TargetFunction(gp, ApproxType.FITC)
    params = np.copy(gp.hyperparams)
#    params = np.log(np.abs(np.random.randn(len(params))))
    print 'likel={0}'.format(target_fct(params))
    print 'gradient={0}'.format(target_fct.gradient(params))
    print 'approx_gradient={0}'.format(spopt.approx_fprime(params, target_fct, np.sqrt(np.finfo(float).eps)))
    


    gp = AMOGPRegression(pKernel, sKernel, rho, likel_fun=PFITCFocusedGaussianLogLikel, infer_method=FocusedExactInference(ApproxType.FITC))
    gp.fit(Xt, yt, Xs, ys, itask)
    print 'likel'
    print gp._log_likel
    yfit = gp.predict(Xp)
    print 'mspe={0}'.format(mspe(yp, yfit))
    print 'sjhwqoj'
    print np.exp(gp.hyperparams)
    plt.plot(Xp,yp,'*')
    plt.plot(Xp, gp.predict(x))
    plt.show()

    target_fct = FocusedExactInference._TargetFunction(gp, ApproxType.PITC)
    params = np.copy(gp.hyperparams)
#    params = np.log(np.abs(np.random.randn(len(params))))
    print 'likel={0}'.format(target_fct(params))
    print 'gradient={0}'.format(target_fct.gradient(params))
    print 'approx_gradient={0}'.format(spopt.approx_fprime(params, target_fct, np.sqrt(np.finfo(float).eps)))
    
    
    gp = FocusedGPRegression(pKernel, sKernel, rho, likel_fun=FTCFocusedGaussianLogLikel, infer_method=FocusedExactInference(ApproxType.PITC))
    gp.fit(Xt, yt, Xs, ys, itask)
    print 'likel'
    print gp._log_likel
    yfit = gp.predict(Xp)
    print 'mspe={0}'.format(mspe(yp, yfit))
    print 'sjhwqoj'
    print np.exp(gp.hyperparams)
    plt.plot(Xp,yp,'*')
    plt.plot(Xp, gp.predict(x))
    plt.show()

    target_fct = FocusedExactInference._TargetFunction(gp, ApproxType.PITC)
    params = np.copy(gp.hyperparams)
#    params = np.log(np.abs(np.random.randn(len(params))))
    print 'likel={0}'.format(target_fct(params))
    print 'gradient={0}'.format(target_fct.gradient(params))
    print 'approx_gradient={0}'.format(spopt.approx_fprime(params, target_fct, np.sqrt(np.finfo(float).eps)))
    
    
    target_fct = FocusedExactInference._TargetFunction(gp, ApproxType.PITC)
    params = np.copy(gp.hyperparams)
    params = np.log(np.abs(np.random.randn(len(params))))
    print 'likel={0}'.format(target_fct(params))
    print 'gradient={0}'.format(target_fct.gradient(params))
    print 'approx_gradient={0}'.format(spopt.approx_fprime(params, target_fct, np.sqrt(np.finfo(float).eps)))
    print np.sqrt(np.finfo(float).eps)
    
    
#    
#    offset = 0
#    for i in xrange(ntasks):
#        print Xs[offset:offset+100].shape
#        print ys[offset:offset+100].shape
#        plt.plot(Xs[offset:offset+100],ys[offset:offset+100], '*')
#        plt.show()
#        offset += 100
#    
#    
#    
#    
#    gp = GPRegression(pKernel, infer_method=ExactInference)
#    gp.fit(Xt, yt)
#    yfit = gp.predict(Xp)
#    print 'mspe={0}'.format(mspe(yp, yfit))
#    
#    
#    plt.plot(Xp,yp,'*')
#    plt.plot(Xp, gp.predict(x))
#    plt.show()
#    
#    pKernel = HiddenKernel(SEKernel(np.log(0.5), np.log(2)) + NoiseKernel(np.log(0.5)))
#    
#    
#    
#    gp = FocusedGPRegression(pKernel, sKernel, rho, likel_fun=SimpleFocusedGaussianLogLikel)
#    gp.fit(Xt, yt, Xs, ys, itask)
#    print 'likel={0}'.format(gp._log_likel)
#    yfit = gp.predict(Xp)
#    print 'mspe={0}'.format(mspe(yp, yfit))
#    print 'sjhwqoj'
#    plt.plot(Xp,yp,'*')
#    plt.plot(Xp, gp.predict(x))
#    plt.show()
#
#
#    
#    gp = FocusedGPRegression(pKernel, sKernel, rho, likel_fun=FocusedGaussianLogLikel, infer_method=FocusedExactInference)
#    #gp = FocusedGPRegression(pKernel, sKernel, rho, likel_fun=FocusedGaussianLogLikel)
#    gp.fit(Xt, yt, Xs, ys, itask)
#    yfit = gp.predict(Xp)
#    print 'mspe={0}'.format(mspe(yp, yfit))
#    print 'Alikel={0}'.format(gp._log_likel)
#    
#    target_fct = FocusedExactInference._TargetFunction(gp)
#    print np.log(np.abs(np.random.randn(5)))
#    
#    plt.plot(Xp,yp,'*')
#    plt.plot(Xp, gp.predict(x))
#    plt.show()
#
#    
#    params = np.copy(gp.hyperparams)
#    #print params[-5:]
#    #params = np.log(np.abs(np.random.randn(23)))
#    
#    print 'params'
#    print params
#    print 'likel={0}'.format(target_fct(params))
#    print 'gradient={0}'.format(target_fct.gradient(params))
#    print 'approx_gradient={0}'.format(spopt.approx_fprime(params, target_fct, np.sqrt(np.finfo(float).eps)))
#    
##    gp = FocusedGPRegression(pKernel, sKernel, rho, likel_fun=FocusedGaussianLogLikel)
##    gp.fit(Xp, yp, Xs, ys, itask)
##    print gp.predict(x[0:5], True)
##    #print yp
##    #print ys
##    
##    gp = GPRegression(pKernel)
##    gp.fit(Xp, yp)
##    print 'single gp'
##    print gp.predict(x[0:5], True)
#    
##    gp = FocusedGPRegression(pKernel, sKernel, rho, infer_method=FocusedExactInference)
##    gp.fit(Xp, yp, Xs, ys, itask)
##    print gp.hyperparams
#    
#    lKernel = DiracConvolvedKernel(SEKernel(np.log(0.5), np.log(2)))
#    #lKernel = CompoundKernel([DiracConvolvedKernel(SEKernel(np.log(0.5), np.log(0.8))), DiracConvolvedKernel(SEKernel(np.log(0.5), np.log(0.8)))])
#    #lKernel = DiracConvolvedKernel(SEKernel(np.log(0.5), np.log(0.8))+NoiseKernel(np.log(0.5)))
#    #lKernel = ExpGaussianKernel([np.log(0.8)])
#    #lKernel = ExpARDSEKernel([np.log(0.8)], np.log(0.4))
#    #lKernel = CompoundKernel([ExpGaussianKernel([np.log(0.8)]), ExpGaussianKernel([np.log(1)])])
#    #pKernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
#    pKernel = NoiseKernel(np.log(0.5))
#    sKernel = ZeroKernel()
#    #theta = np.array([np.log(0.5), np.log(1)])
#    theta = np.array([np.log(1)])
#    print type(lKernel)
#    print type(sKernel)
#    print type(DiracConvolvedKernel(SEKernel(np.log(1), np.log(1))))
#    print lKernel.ntheta
#    
#    amtlKernel = DiracConvolvedAMTLKernel(lKernel, theta, ntasks, pKernel, sKernel)
#    
#    gp = FMOGPRegression(amtlKernel)
#    likel = gp.fit(Xt, yt, Xs, ys, itask)
#    print 'Alikel1={0}'.format(gp._log_likel)
#    print gp._likel_fun.gradient()
#    
#    plt.plot(Xp,yp,'*')
#    plt.plot(Xp, gp.predict(x))
#    plt.show()
#    
#
#    target_fct = FMOGPExactInference._TargetFunction(gp)
#    
#    params = np.copy(amtlKernel.params)
#    #params = np.log(np.abs(np.random.randn(len(params))))
#    #params[-5:] = np.log(0.5)
#    print 'likel={0}'.format(target_fct(params))
#    print 'gradient={0}'.format(target_fct.gradient(params))
#    print 'approx_gradient={0}'.format(spopt.approx_fprime(params, target_fct, np.sqrt(np.finfo(float).eps)))
#
#    gp = FMOGPRegression(amtlKernel, infer_method=FMOGPExactInference)
#    likel = gp.fit(Xt, yt, Xs, ys, itask)
#    print 'likel1={0}'.format(gp._log_likel)
#    print gp._likel_fun.gradient()
#    print 'params={0}'.format(amtlKernel.params)
#    print 'params={0}'.format(amtlKernel._theta)
#    
#    
#    plt.plot(Xp,yp,'*')
#    plt.plot(Xp, gp.predict(x))
#    plt.show()
#    
#    target_fct = FMOGPExactInference._TargetFunction(gp)
#    
#    params = np.copy(amtlKernel.params)
#    params = np.log(np.abs(np.random.randn(len(params))))
#    #params[-5:] = np.log(0.5)
#    print 'likel={0}'.format(target_fct(params))
#    print 'gradient={0}'.format(target_fct.gradient(params))
#    print 'approx_gradient={0}'.format(spopt.approx_fprime(params, target_fct, np.sqrt(np.finfo(float).eps)))
#
#    
#    amtlKernel = DiracConvolvedAMTLKernel(lKernel, theta, ntasks, pKernel, sKernel)
#    print 'params={0}'.format(amtlKernel.params)