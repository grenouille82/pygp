'''
Created on Sep 25, 2012

@author: marcel
'''
import numpy as np
import scipy.optimize as spopt
import time

from upgeo.base.infer import InferenceAlgorithm
from scipy.linalg.decomp_cholesky import cho_solve
from upgeo.util.math import fmin_ras, fmin_ern
from numpy.linalg.linalg import LinAlgError
from upgeo.util.glob import APPROX_TYPE


class CMOGPOnePassInference(InferenceAlgorithm):
    
    __slots__ = ()
    
    def apply(self, gp):
        #training cases
        X = gp._X
        y = gp._y
        
        itask = gp._itask
        kernel = gp._kernel
        likel_fun = gp._likel_fun
        n = gp._n

        #determine the cov matrix K and compute lower matrix L        
        #K = kernel.cov_old(X, itask)
        K = kernel.cov(X, itaskX=itask)        
        L = np.linalg.cholesky(K)
        
        #np.linalg.chole
        
        iK = cho_solve((L,1), np.eye(n))
                
        #compute alpha = K^(-1)*y 
        alpha = cho_solve((L,1), y)
        
        gp._K = K
        gp._L = L
        gp._iK = iK
        gp._alpha = alpha
        
        
        #compute the model log likelihood (gp relevant parameters must be updated before)
        #todo: check if it logical to update the likel here        
        likel = likel_fun()
        gp._log_likel = likel
        
class CMOGPExactInference(InferenceAlgorithm):
    
    __slots__ = ()

    def apply(self, gp):
        '''
        '''
        CMOGPExactInference._update_model(gp)
        
        params = gp.hyperparams
        
        f = CMOGPExactInference._TargetFunction(gp) #reference to fun to be optimized
        fprime = f.gradient                     #reference to the gradient
        cb = f.callback                         #reference to the callback fun
        x0 = np.copy(params)                    #initial guess of the hyperparameters
       
        
        #opt_result = spopt.fmin_bfgs(f, x0, fprime, callback=cb, gtol=1e-12, maxiter=300, full_output = True)
        #opt_result = spopt.fmin_bfgs(f, x0, fprime, callback=cb, full_output = True)
        #opt_result = spopt.fmin_cg(f, x0, fprime, gtol=1e-2, callback=cb, full_output = True)
        opt_result = fmin_ras(x0, f, fprime, [], 100)
        xopt = opt_result[0]    #optimal hyperparameters with respect to log likelihood
        #xopt = fmin_ern(f, x0, fprime, maxiter=50)
        
        #Debug output
        if False:
            fun_calls = opt_result[4]
            grad_calls = opt_result[5]
            print 'number of opt calls: fun={0}, grad={1}'.format(fun_calls, grad_calls)
            
            err = spopt.check_grad(f, fprime, x0)
            print 'gradient check at {0}: error={1}'.format(x0, err)
            print spopt.approx_fprime(x0, f, np.sqrt(np.finfo(float).eps))
            
        
        CMOGPExactInference._update_parameters(gp, xopt)
        CMOGPExactInference._update_model(gp)
            
    class _TargetFunction(object):
        
        __slots__ = ('_gp',
                     '_likel_fun'
                     )
        
        def __init__(self, gp):
            self._gp = gp
            self._likel_fun = gp._likel_fun
        
        def __call__(self, params):
            flag = np.logical_or(np.isnan(params), np.isinf(params))
            if np.any(flag):
                print 'bad params={0}'.format(params)
                return 1e+300

            
            gp = self._gp
            old_params = gp.hyperparams 
            
            try: 
                CMOGPExactInference._update_parameters(gp, params)
                CMOGPExactInference._update_model(gp)
            except LinAlgError:
                CMOGPExactInference._update_parameters(gp, old_params)
                return 1e+300
            
            #print 'gp likel={0}'.format(-gp._log_likel)
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
            
            CMOGPExactInference._update_parameters(gp, params)
            CMOGPExactInference._update_model(gp)
            

    @staticmethod
    def _update_parameters(gp, params):
        kernel = gp._kernel
        kernel.params = params
        
    @staticmethod
    def _update_model(gp):
        #training cases
        X = gp._X
        y = gp._y
        
        itask = gp._itask
        kernel = gp._kernel
        likel_fun = gp._likel_fun
        n = gp._n

        #determine the cov matrix K and compute lower matrix L        
        #K = kernel.cov_old(X, itask)
        K = kernel.cov(X, itaskX=itask)        
        L = np.linalg.cholesky(K)
        
        #np.linalg.chole
        
        iK = cho_solve((L,1), np.eye(n))
        
        #compute alpha = K^(-1)*y 
        alpha = cho_solve((L,1), y)
        
        gp._K = K
        gp._L = L
        gp._iK = iK
        gp._alpha = alpha
        
        
        #compute the model log likelihood (gp relevant parameters must be updated before)
        #todo: check if it logical to update the likel here        
        likel = likel_fun()
        gp._log_likel = likel

class SparseCMOGPOnePassInference(InferenceAlgorithm):
    
    __slots__ = ()
    
    def apply(self, gp):
        #training cases + inducing points
        X = gp._X
        y = gp._y
        Xu = gp._Xu
        

        n = gp._n
        m = len(Xu)
        
        itask = gp._itask
        task_sizes = gp._task_sizes        
        k = gp._ntasks

        kernel = gp._kernel
        beta = gp._beta
        likel_fun = gp._likel_fun
        approx = gp._approx_type

        #determine the cov matrix K and compute lower matrix L        
        #K = kernel.cov_old(X, itask)
        Ku = kernel.cov(Xu)  #add some jitter
        Lu = np.linalg.cholesky(Ku + np.eye(m)*1e-12)
        iKu = cho_solve((Lu,1), np.eye(m))
        #print 'Ku={0}'.format(Ku)
        Kfu = kernel.cov(X, Xu, itask)
        V1 = np.linalg.solve(Lu, Kfu.T) #V=Luu^-1*Kuf
        
        #print 'Kfu*iKu*Kfu'
        #print np.dot(V1.T, V1)
        #print np.dot(np.dot(Kfu, iKu), Kfu.T)  
        #print np.dot(np.dot(Kfu, iKu), Kfu.T)-np.dot(V1.T, V1)
        
        
        if approx == APPROX_TYPE.FITC:
            #Z = np.dot(np.dot(Kfu, iKu), Kfu.T) 
            Kf = kernel.cov(X, itaskX=itask, diag=True)
            D = Kf - np.sum(V1*V1,0)
            D = D + 1.0/np.repeat(D, task_sizes) if beta != None else D
            Ld = np.sqrt(D)             
            iD = 1.0/D
            iDy = y*iD
            
            #print 'Ddiff'
            #print D-np.diag(kernel.cov(X, itaskX=itask)-Z)
            
            r = np.dot(Kfu.T, iDy)
            
            V2 = V1/Ld
            V3 = np.linalg.cholesky(np.eye(m)+np.dot(V2, V2.T)) #V3 = chol(I + V*V)
            La = np.dot(Lu, V3)
            A = np.dot(La, La.T) #could be removed, if not necessary
            
            #print 'Adiff'
            #print A - (Ku + np.dot(np.dot(Kfu.T, np.diag(iD)), Kfu))
            #print A
            #print (Ku + np.dot(np.dot(Kfu.T, np.diag(iD)), Kfu))
        elif approx == APPROX_TYPE.PITC:
            itask = np.r_[itask, n]
            
            Kf = np.empty(k, dtype=object)
            D = np.empty(k, dtype=object)
            Ld = np.empty(k, dtype=object)
            iD = np.empty(k, dtype=object)
            iDy = np.zeros(n)
            U = np.zeros((m,m))
            r = np.zeros(m)
            for i in xrange(k):
                start = itask[i]
                end = itask[i+1]
                
                Kf[i] = kernel.cov_block(X[start:end], i=i)
                #print 'Kfi={0}'.format(Kf[i])
                D[i] = Kf[i] - np.dot(V1[:,start:end].T, V1[:,start:end])
                if beta != None:
                    D[i] = D[i] + np.eye(end-start)/beta[i]
                
                Ld[i] = np.linalg.cholesky(D[i])                
                iD[i] = cho_solve((Ld[i], 1), np.eye(end-start))
                iDy[start:end] = np.dot(iD[i], y[start: end])
                
                U += np.dot(np.dot(Kfu[start:end].T, iD[i]), Kfu[start:end])
                r += np.dot(Kfu[start:end].T, iDy[start:end])
            
            A = Ku + U
            La = np.linalg.cholesky(A+np.eye(m)*1.e-12)
        else:
            raise TypeError('Unknown approx method')
        
        #print 'r={0}'.format(r)    
        #print np.dot(Kfu.T, iDy)
        
        iA = cho_solve((La, 1), np.eye(m))

        #Lz = np.linalg.cholesky(Ku + np.dot(np.dot(Kfu.T, np.diag(iD)), Kfu)+np.eye(m)*1e-12)        
        #print 'iA infer'
        #print iA - np.linalg.inv(A+1e-12*np.eye(m))
        #print np.linalg.inv(A+1e-12*np.eye(m))-cho_solve((Lz, 1), np.eye(m))
        #print np.linalg.inv(Ku + np.dot(np.dot(Kfu.T, np.diag(iD)), Kfu)) -np.linalg.inv(A+1e-12*np.eye(m))
        #print iA-np.linalg.inv(Ku + np.dot(np.dot(Kfu.T, np.diag(iD)), Kfu))
        #print iA-cho_solve((Lz, 1), np.eye(m))
        #print A-(Ku + np.dot(np.dot(Kfu.T, np.diag(iD)), Kfu))
        #print La-Lz
        #print cho_solve((La, 1), np.eye(m))-cho_solve((Lz, 1), np.eye(m))
        
            
        alpha = np.dot(iA, r)
        
        gp._Ku = Ku
        gp._Lu = Lu 
        gp._iKu = iKu
        gp._Kf = Kf
        gp._Kfu = Kfu
        gp._D = D
        gp._Ld = Ld
        gp._iD = iD
        gp._iDy = iDy
        gp._A = A
        gp._iA = iA
        gp._La = La
        
        #gp._U = U
        gp._r = r
        
        gp._alpha = alpha
        
        #compute the model log likelihood (gp relevant parameters must be updated before)
        #todo: check if it logical to update the likel here        
        likel = likel_fun()
        gp._log_likel = likel
        
class SparseCMOGPExactInference(InferenceAlgorithm):
    
    __slots__ = ()

    def apply(self, gp):
        '''
        '''
        SparseCMOGPExactInference._update_model(gp)
        
        params = gp.hyperparams
        if not gp._fix_inducing:
            params = np.r_[params, gp._Xu.flatten()]
        if gp._beta != None:
            params = np.r_[params, gp._beta]
        
        f = SparseCMOGPExactInference._TargetFunction(gp) #reference to fun to be optimized
        fprime = f.gradient                     #reference to the gradient
        cb = f.callback                         #reference to the callback fun
        x0 = np.copy(params)                    #initial guess of the hyperparameters
       
        
        opt_result = spopt.fmin_bfgs(f, x0, fprime, callback=cb, gtol=1e-12, maxiter=300, full_output = True)
        #opt_result = spopt.fmin_bfgs(f, x0, fprime, callback=cb, gtol=1e-12, full_output = True, maxiter=200)
        #opt_result = spopt.fmin_cg(f, x0, fprime, gtol=1e-2, callback=cb, full_output = True)
        #opt_result = fmin_ras(x0, f, fprime, [], 300)
        xopt = opt_result[0]    #optimal hyperparameters with respect to log likelihood
        
        #print 'warnflag={0}'.format(opt_result[6])
        print 'gopt={0}'.format(opt_result[2])
        #xopt = fmin_ern(f, x0, fprime, maxiter=50)
        #xopt = spopt.approx_fprime(params, f, np.sqrt(np.finfo(float).eps))
        
        #Debug output
        if False:
            fun_calls = opt_result[4]
            grad_calls = opt_result[5]
            print 'number of opt calls: fun={0}, grad={1}'.format(fun_calls, grad_calls)
            
            err = spopt.check_grad(f, fprime, x0)
            print 'gradient check at {0}: error={1}'.format(x0, err)
            print spopt.approx_fprime(x0, f, np.sqrt(np.finfo(float).eps))
            
        
        SparseCMOGPExactInference._update_parameters(gp, xopt)
        SparseCMOGPExactInference._update_model(gp)
            
    class _TargetFunction(object):
        
        __slots__ = ('_gp',
                     '_likel_fun'
                     )
        
        def __init__(self, gp):
            self._gp = gp
            self._likel_fun = gp._likel_fun
        
        def __call__(self, params):
            flag = np.logical_or(np.isnan(params), np.isinf(params))
            if np.any(flag):
                print 'bad params={0}'.format(params)
                return 1e+300

            
            gp = self._gp
            old_params = gp.hyperparams 
            if not gp._fix_inducing:
                old_params = np.r_[old_params, gp._Xu.flatten()]
            if gp._beta != None:
                old_params = np.r_[old_params, gp._beta]
            try: 
                SparseCMOGPExactInference._update_parameters(gp, params)
                SparseCMOGPExactInference._update_model(gp)
            except LinAlgError:
                SparseCMOGPExactInference._update_parameters(gp, old_params)
                return 1e+300
            
            #print 'gp likel={0}'.format(-gp._log_likel)
            #print 'gp likel params ={0}'.format(np.exp(params))
                
            return -gp._log_likel
        
        def gradient(self, params):
            #print 'params1={0}'.format(params)
            likel_fun = self._likel_fun
            #print 'gradient={0}'.format(-likel_fun.gradient()) 
            t = time.time()
            grad = -likel_fun.gradient()
            #print 'grad_time={0}'.format(time.time()-t)
            flag = np.logical_or(np.isnan(grad), np.isinf(grad))
            if np.any(flag):
                grad[flag] = 0
                print 'bad gradient={0}'.format(grad)
            return grad
                
        def callback(self, params):
            gp = self._gp
            
            SparseCMOGPExactInference._update_parameters(gp, params)
            SparseCMOGPExactInference._update_model(gp)
            

    @staticmethod
    def _update_parameters(gp, params):
        kernel = gp._kernel
        if gp._fix_inducing:
            kernel.params = params[:kernel.nparams]
            offset = kernel.nparams
        else:
            Xu = gp._Xu
            m,d = Xu.shape
            kernel.params = params[0:kernel.nparams]
            offset = kernel.nparams
            Xu = params[offset:offset+m*d].reshape(m,d)
            gp._Xu = Xu
            offset = offset + m*d
        
        if gp._beta != None:
            gp._beta = params[offset:]
        
    @staticmethod
    def _update_model(gp):
        #training cases + inducing points
        t = time.time()
        X = gp._X
        y = gp._y
        Xu = gp._Xu

        n = gp._n
        m = len(Xu)
        
        itask = gp._itask
        task_sizes = gp._task_sizes        
        k = gp._ntasks

        kernel = gp._kernel
        beta = gp._beta
        likel_fun = gp._likel_fun
        approx = gp._approx_type

        #determine the cov matrix K and compute lower matrix L        
        #K = kernel.cov_old(X, itask)
        Ku = kernel.cov(Xu)  #add some jitter
        Lu = np.linalg.cholesky(Ku + np.eye(m)*1e-12)
        iKu = cho_solve((Lu,1), np.eye(m))
        Kfu = kernel.cov(X, Xu, itask)
        V1 = np.linalg.solve(Lu, Kfu.T) #V=Luu^-1*Kuf
        
        
        if approx == APPROX_TYPE.FITC:
            Kf = kernel.cov(X, itaskX=itask, diag=True)
            D = Kf - np.sum(V1*V1,0)
            D = D + 1.0/np.repeat(D, task_sizes) if beta != None else D
            
            Ld = np.sqrt(D)             
            iD = 1.0/D
            iDy = y*iD
            
            
            r = np.dot(Kfu.T, iDy)
            
            V2 = V1/Ld
            V3 = np.linalg.cholesky(np.eye(m)+np.dot(V2, V2.T)) #V3 = chol(I + V*V)
            La = np.dot(Lu, V3)
            A = np.dot(La, La.T) #could be removed, if not necessary
            
        
            
        elif approx == APPROX_TYPE.PITC:
            itask = np.r_[itask, n]
            
            Kf = np.empty(k, dtype=object)
            D = np.empty(k, dtype=object)
            Ld = np.empty(k, dtype=object)
            iD = np.empty(k, dtype=object)
            iDy = np.zeros(n)
            U = np.zeros((m,m))
            r = np.zeros(m)
            for i in xrange(k):
                start = itask[i]
                end = itask[i+1]
                
                Kf[i] = kernel.cov_block(X[start:end], i=i)
                D[i] = Kf[i] - np.dot(V1[:,start:end].T, V1[:,start:end])
                if beta != None:
                    D[i] = D[i] + np.eye(end-start)/beta[i]
                
                Ld[i] = np.linalg.cholesky(D[i])                
                iD[i] = cho_solve((Ld[i], 1), np.eye(end-start))
                iDy[start:end] = np.dot(iD[i], y[start: end])
                
                U += np.dot(np.dot(Kfu[start:end].T, iD[i]), Kfu[start:end])
                r += np.dot(Kfu[start:end].T, iDy[start:end])
            A = Ku + U
            La = np.linalg.cholesky(A+1e-12*np.eye(m))
        else:
            raise TypeError('Unknown approx method')
            
        
        iA = cho_solve((La, 1), np.eye(m))    
        alpha = np.dot(iA, r)
        
        gp._Ku = Ku
        gp._Lu = Lu
        gp._iKu = iKu
        gp._Kf = Kf
        gp._Kfu = Kfu
        gp._D = D
        gp._Ld = Ld
        gp._iD = iD
        gp._iDy = iDy
        gp._A = A
        gp._iA = iA
        gp._La = La
        
        #gp._U = U
        gp._r = r
        
        gp._alpha = alpha
        
        #compute the model log likelihood (gp relevant parameters must be updated before)
        #todo: check if it logical to update the likel here        
        likel = likel_fun()
        gp._log_likel = likel

        #print 'update_time={0}'.format(time.time()-t)