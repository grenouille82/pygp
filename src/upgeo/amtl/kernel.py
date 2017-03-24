'''
Created on Jun 9, 2012

@author: marcel
'''
import numpy as np
from upgeo.base.kernel import NoiseKernel, ARDSEKernel, CompoundKernel,\
    ExpARDSEKernel

        
class AMTLKernel(object):
    __slots__ = () 
    
    def cov(self, Xp, Xs=None):
        '''
        '''
        
    def cross_cov(self):
        '''
        '''
    

class DiracConvolvedAMTLKernel(object):
    '''
    @todo: - maybe refactor the cov and cross_cov function because the are overloaded by the parameters 
            (its hard to understand, what does the function do). so it could be better to seperate each 
            function for independent and latent kernels.
    '''
    
    __slots__ = ('_lKernel',    #latent kernel
                 '_pKernel',    #private kernel for the primary tasks
                 '_sKernel',    #vector of private kernels for the secondary tasks)
                 '_theta',      
                 '_ntasks',     #number of secondary tasks  
                 '_nparams'           
                 )
    
    def __init__(self, lKernel, theta, ntasks, pKernel=None, sKernel=None):
        '''
        Parameters:
            latent_kernel - underlying kernel which controls the correlation 
                            between primary and secondary tasks (internally duplicated q times)
            q             - the number of latent kernels
            pKernel       - the private kernel for the primary task
            sKernel       - the private kernel for the secondary tasks ((internally duplicated ntasks times)
            theta         - initial parameters for the smoothing kernel to latent kernel
            ntasks        - number of background tasks
        '''
        theta = np.asarray(theta).ravel()
        if lKernel.ntheta != len(theta):
            raise ValueError('size of theta is not equal to the expectect of the latent kernel')
        if ntasks < 1:
            raise ValueError('number of secondary tasks must be at least 1')
        
        self._lKernel = lKernel
        self._pKernel = pKernel
        self._ntasks = ntasks
        self._sKernel = np.empty(ntasks, dtype='object')
        self._nparams = lKernel.nparams+lKernel.ntheta*ntasks+pKernel.nparams+sKernel.nparams*ntasks 
        for i in xrange(ntasks):
            self._sKernel[i] = sKernel.copy()
        self._theta = theta[np.newaxis,:].repeat(ntasks, 0)
     
    def cov(self, X, task=0, latent=True, diag=False):
        '''
            Parameters:
                X    - data matrix for which the covariance matrix is computed
                Z    - if Z is specified then the covariance matrix between X 
                       and Z is returned
                task - specifies the task from which the data comes and for 
                       which the covariance matrix is computed. Task 0 indicates
                       the primary covariance matrix, and with the values between
                       [1,ntasks] the covaraince matrix for the specific task is
                       computed
                latent - specify whether the  covariance matrix over the latent or
                         independent kernel is computed
                diag:  - if true, the diagonal covariance matrix is returned, but 
                         only in the case if Z is not specified
            Cases:
                cov[fp(x), fp(X)] -> task = 0 and latent = true and Z = None
                cov[fs(x), fs(X)] -> task = 1..ntasks and latent = true and Z = None
                cov[gp(x), gp(X)] -> task = 0 and latent = False and Z = None
                cov[gs(x), gs(X)] -> task = 1..ntasks and latent = False and Z = None
                
                
                
        '''
        if task == 0:
            #primary tasks
            if latent:
                lKernel = self._lKernel
                K = lKernel.latent_cov(X, diag=diag)
            else:
                n = len(X)
                pKernel = self._pKernel
                K = pKernel(X, diag=diag) if pKernel != None else np.zeros((n,n))
        else:
            if task > self._ntasks:
                raise ValueError('unknown task: {0}'.format(task))
         
            
            theta = self._theta[task-1]
            if latent:
                #thats is problematic, because we cannot easily compute its gradient wrt to the latent parameters
                lKernel = self._lKernel
                K = lKernel.cov(X, theta, diag) 
            else:
                n = len(X)
                sKernel = self._sKernel[task-1]
                K = sKernel(X, diag=diag) if sKernel != None else np.zeros((n,n))
        return K
    
    def cross_cov(self, X, Z, task=0, latent=True):
        '''
        '''
        if task == 0:
            if latent:
                lKernel = self._lKernel
                K = lKernel.latent_cov(X, Z) #problematic
            else:
                n = len(X)
                m = len(Z)
                pKernel = self._pKernel
                K = pKernel(X, Z) if pKernel != None else np.zeros((n,m))
        else:
            if task > self._ntasks:
                raise ValueError('unknown task: {0}'.format(task))
            
            if latent:
                #thats is problematic, because we cannot easily compute its gradient wrt to the latent parameters
                lKernel = self._lKernel
                theta = self._theta[task-1]
                K = lKernel.cross_cov(X, Z, theta, latent=True)
            else:
                n = len(X)
                m = len(Z)
                sKernel = self._sKernel[task-1]
                K = sKernel(X, Z) if sKernel != None else np.zeros((n,m))
        return K
        
    def derivate(self, task=0, latent=True):
        
        class LatentDerivativeFunc(object):
            '''
                Derivate of the latent kernel
            '''
            def __init__(self, kernel, task, theta=None):
                self.kernel = kernel #the latent kernel
                self.task = task
                self.theta = theta
                self.nparams = kernel.nparams if task == 0 else kernel.ntheta
                
            def cov(self, X, i, q=None, diag=False):
                if i >= self.nparams:
                    raise ValueError('Unknown hyperparameter')
                
                task = self.task
                kernel = self.kernel
                
                if task == 0:
                    #primary task
                    dKernel = kernel.derivate(i)
                    if q == None:
                        dK = dKernel.latent_cov(X, diag=diag)
                    else:
                        #its a hack to compute the derivate lKernel.cov(X, theta, diag) 
                        theta = self.theta[q-1]
                        dK = dKernel.cov(X, theta, diag=diag)
                else:
                    theta = self.theta[task-1]
                    dKernel = kernel.derivateTheta(i)
                    dK = dKernel.cov(X, theta, diag)
                return dK
                
            def cross_cov(self, X, Z, i, q=None):
                if i >= self.nparams:
                    raise ValueError('Unknown hyperparameter')

                task = self.task
                kernel = self.kernel
                if task == 0:
                    dKernel = kernel.derivate(i)
                    if q == None:
                        dK = dKernel.latent_cov(X, Z)
                    else:
                        #its a hack to compute the derivate lKernel.cov(X, theta, diag)
                        theta = self.theta[q-1]
                        dK = dKernel.cross_cov(X, Z, theta, latent=True)
                else:
                    theta = self.theta[task-1]
                    dKernel = kernel.derivateTheta(i)
                    dK = dKernel.cross_cov(X, Z, theta, latent=True)
                return dK
                    
                
        class PrivateDerivateFunc(object):
            
            def __init__(self, kernel):
                '''
                '''
                self.kernel = kernel
                self.nparams = kernel.nparams if kernel != None else 0
                
            def cov(self, X, i, diag=False):
                
                if i >= self.nparams:
                    raise ValueError('Unknown hyperparameter')
                
                kernel = self.kernel
                dKernel = kernel.derivate(i)
                dK = dKernel(X, diag=diag)
                return dK
            
            def cross_cov(self, X, Z, i, diag=False):
                '''
                TODO: throw an exception if i is out of range
                Returns an empty gradient, because no correlation between private components
                '''
                if i >= self.nparams:
                    raise ValueError('Unknown hyperparameter')

                kernel = self.kernel
                dKernel = kernel.derivate(i)
                dK = dKernel(X, Z)
                return dK
        
        if latent:
            '''
            '''
            if task > self._ntasks:
                raise ValueError('unknown task: {0}'.format(task))
            kernel = self._lKernel
            theta = self._theta
            deriv_fct = LatentDerivativeFunc(kernel, task, theta)
        else:
            if task == 0:
                kernel = self._pKernel
            else:
                if task > self._ntasks:
                    raise ValueError('unknown task: {0}'.format(task))
         
                kernel = self._sKernel[task-1]
            deriv_fct = PrivateDerivateFunc(kernel)
        return deriv_fct
        
    def _get_params(self):
        lKernel = self._lKernel
        pKernel = self._pKernel
        sKernel = self._sKernel
        theta = self._theta
        
        ntasks = self._ntasks
        
        
        params = np.r_[lKernel.params, theta.ravel()]
        params = np.r_[params, pKernel.params] if pKernel != None else params
        if sKernel != None:
            for i in xrange(ntasks):
                params = np.r_[params, sKernel[i].params]
            
        return params
    
    def _set_params(self, params):
        lKernel = self._lKernel
        pKernel = self._pKernel
        sKernel = self._sKernel
        
        ntasks = self._ntasks
        ntheta = lKernel.ntheta
        
        lKernel.params = params[0:lKernel.nparams]
        offset = lKernel.nparams
        self._theta = np.reshape(params[offset:offset+ntasks*ntheta], (ntasks, ntheta))
        offset += ntasks*ntheta
        if pKernel != None:
            pKernel.params = params[offset:offset+pKernel.nparams]
        offset += pKernel.nparams
        if sKernel != None:
            for i in xrange(ntasks):
                sKernel[i].params = params[offset:offset+sKernel[i].nparams]
                offset += sKernel[i].nparams
        
        
         
    params = property(fget=_get_params, fset=_set_params)
    
    def _get_nparams(self):
        return self._nparams
    
    nparams = property(fget=_get_nparams)
    
    def latent_kernel(self):
        return self._lKernel
    
    def private_kernel(self, task):
        kernel = None
        if task == 0:
            kernel = self._pKernel
        else:
            if task > self._ntasks:
                raise ValueError('unknown task: {0}'.format(task))
            kernel = self._sKernel[task-1]
        return kernel
                
        
class SmoothConvolvedAMTLKernel():
    
    __slots__ = ('_lKernel',    #latent kernel
                 '_pKernel',    #private kernel for the primary tasks
                 '_sKernel',    #vector of private kernels for the secondary tasks)
                 '_theta',      
                 '_ntasks'     #number of secondary tasks             
                 )
    
    def __init__(self, lKernel, thetaP, thetaS, ntasks, pKernel=None, sKernel=None):
        '''
        Parameters:
            latent_kernel - underlying kernel which controls the correlation 
                            between primary and secondary tasks (internally duplicated q times)
            q             - the number of latent kernels
            pKernel       - the private kernel for the primary task
            sKernel       - the private kernel for the secondary tasks ((internally duplicated ntasks times)
            theta         - initial parameters for the smoothing kernel to latent kernel
            ntasks        - number of background tasks
        '''
        theta = np.asarray(theta).ravel()
        if lKernel.ntheta != len(theta):
            raise ValueError('size of theta is not equal to the expectect of the latent kernel')
        if ntasks < 1:
            raise ValueError('number of secondary tasks must be at least 1')
        
        self._lKernel = lKernel
        self._pKernel = pKernel
        self._ntasks = ntasks
        self._sKernel = np.empty(ntasks, dtype='object')
        for i in xrange(ntasks):
            self._sKernel[i] = sKernel.copy()
        self._theta = theta[np.newaxis,:].repeat(ntasks, 0)
     
    def cov(self, X, task, latent=True, diag=False):
        '''
        Problems: - add the private kernel for secondary tasks
                  - how we can incoparate that we only consider the cov matrix of latent kernels or private kernels
        '''

        if task == 0:
            #primary tasks
            if latent:
                lKernel = self._lKernel
                thetaP = self._thetaP
                K = lKernel.cov(X, thetaP, diag) 
            else:
                pKernel = self._pKernel
                K = pKernel(X, diag) if pKernel != None else np.zeros(X.shape)
        else:
            if task > self._ntasks:
                raise ValueError('unknown task: {0}'.format(task))
            
            if latent:
                thetaS = self._thetaS[task-1]
                lKernel = self._lKernel
                K = lKernel.cov(X, thetaS, diag)
            else:
                sKernel = self._sKernel[task-1]
                K = sKernel(X, diag) if sKernel != None else np.zeros(X.shape)
        return K

    def cross_cov(self, X, Z, task=0):
        '''
        '''
        lKernel = self._lKernel
        thetaP = self._thetaP
        if task == 0:
            K = lKernel.cross_cov(X, Z, thetaP)
        else:
            if task > self._ntasks:
                raise ValueError('unknown task: {0}'.format(task))
             
            lKernel = self._lkernel
            thetaS = self._thetaS[task-1]
            
            K = lKernel.cross_cov(X, Z, thetaP, thetaS, latent=True)
        return K
    
def __test_mtl_cov(kernel, Xp, Xs):
    print 'convolved covariance matrix:'
    print kernel.cov(Xp, task=0)
    print kernel.cov(Xs, task=1)
    print kernel.cov(Xp, task=0, diag=True)
    print 'independent covariance matrix:'
    print kernel.cov(Xp, latent=False, task=0)
    print kernel.cov(Xs, latent=False, task=1)
    print kernel.cov(Xp, latent=False, task=0, diag=True)
    
    print 'latent cross covariance matrix:'
    print kernel.cross_cov(Xp, Xs)
    print kernel.cross_cov(Xp, Xs, latent=False)
    print kernel.cross_cov(Xp, Xs, task=1)
    print kernel.cross_cov(Xp, Xs, task=1, latent=False)
    

if __name__ == '__main__':
    Xp =  np.array( [[-0.5046,    0.3999,   -0.5607],
                     [-1.2706,   -0.9300,    2.1778],
                     [-0.3826,   -0.1768,    1.1385],
                     [0.6487,   -2.1321,   -2.4969],
                     [0.8257,    1.1454,    0.4413],
                     [-1.0149,   -0.6291,   -1.3981],
                     [-0.4711,   -1.2038,   -0.2551],
                     [0.1370,   -0.2539,    0.1644],
                     [-0.2919,   -1.4286,    0.7477],
                     [0.3018,   -0.0209,   -0.2730]])
    
    Xs =  np.array( [[0.5046,     1.5999,    2.5607],
                     [-1.2706,   -0.9300,    0.4778],
                     [1.2919,   1.4286,      0.6477],
                     [0.2018,   -0.08279,   -0.8730]])
    
    Y = np.random.randn(10,8)
    
    lKernel = CompoundKernel([ExpARDSEKernel(np.log(1)*np.ones(3), np.log(2)), ExpARDSEKernel(np.log(0.7)*np.ones(3), np.log(0.1))])
    pKernel = NoiseKernel(np.log(1))
    sKernel = ARDSEKernel(np.log(5)*np.ones(3), np.log(1)) + NoiseKernel(np.log(0.5))
    
    theta = np.r_[np.log(1)*np.ones(3), np.log(0.1), np.log(1)*np.ones(3), np.log(0.1)]
    ntasks = 2
    
    mtlKernel = DiracConvolvedAMTLKernel(lKernel, theta, ntasks, pKernel, sKernel)
    __test_mtl_cov(mtlKernel, Xp, Xs)
    
    print 'beforeparams={0}'.format(mtlKernel.params)
    mtlKernel.params = np.log(np.abs(np.random.randn(mtlKernel.nparams)))
    print 'afterparams={0}'.format(mtlKernel.params)
    #__test_mtl_cov(mtlKernel, Xp, Xs)
    
    mtlKernel._get_params()
    mtlKernel._set_params(np.log(np.abs(np.random.randn(mtlKernel.nparams))))
    print (mtlKernel._get_params())
    __test_mtl_cov(mtlKernel, Xp, Xs)
    
    