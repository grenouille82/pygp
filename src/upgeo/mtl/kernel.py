'''
Created on Sep 25, 2012

@author: marcel
'''

import numpy as np

class MTLKernel(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
    
    def __call__(self):
        '''    
        '''
        

class ConvolvedMTLKernel(object):
    '''
    TODO: - seperate noise and idp kernel
          - allow to specify for each tasks indivually wheter the idp kernel is specified or not
    '''
    
    __slots__ = ('_dpKernel',   #dependent kernel is shared between all tasks
                 '_idpKernel',  #independent kernel reflects the individual behaviour of each process
                 '_theta',
                 '_ntasks',     #number of tasks
                 '_nparams',    #total number of hyperparams
                 )
    
    def __init__(self, dpKernel, theta, ntasks, idpKernel):
        '''
        Parameters:
        '''
        theta = np.asarray(theta).ravel()
        if dpKernel.ntheta != len(theta):
            raise ValueError('size of theta is not equal to the expectect of the dependent kernel')
        if ntasks < 1:
            raise ValueError('number tasks must be at least 1')
        
        self._dpKernel = dpKernel
        self._ntasks = ntasks
        self._idpKernel = np.empty(ntasks, dtype='object')
        self._nparams = dpKernel.nparams+dpKernel.ntheta*ntasks+idpKernel.nparams*ntasks 
        for i in xrange(ntasks):
            self._idpKernel[i] = idpKernel.copy()
        self._theta = theta[np.newaxis,:].repeat(ntasks, 0)
    
    
    def cov(self, X, Z=None, itaskX=None, itaskZ=None, q=None, diag=False):
        '''
        '''
        xeqz = Z == None
        n = X.shape[0]
        
        dpKernel = self._dpKernel
        ntasks = self._ntasks
        if (xeqz and itaskX == None) or (itaskX == None and itaskZ == None):
            '''
            No block(task) indices are specified, therefore the covariance
            of the latent process is computed.
            '''
            K = dpKernel.lcov(X, Z, diag=diag)
            
        elif xeqz:
            '''
            Compute full covariance matrix for X
            '''
            if len(itaskX) != ntasks:
                raise ValueError('Insufficent number of task indices.')
            
            itaskX = np.r_[itaskX, n]
            if diag:
                K = np.zeros(n)
                for i in xrange(ntasks):
                    start = itaskX[i]
                    end = itaskX[i+1]
                    K[start:end] = self.cov_block(X[start:end,:], i=i, diag=True)
            else: 
                K = np.zeros((n,n))
                for i in xrange(ntasks):
                    starti = itaskX[i]
                    endi = itaskX[i+1]
                    Xi = X[starti:endi,:]
                    K[starti:endi, starti:endi] = self.cov_block(Xi, i=i)
                    for j in xrange(i+1, ntasks):
                        startj = itaskX[j]
                        endj = itaskX[j+1]
                        Xj = X[startj:endj,:]
                        
                        K[starti:endi, startj:endj] = self.cov_block(Xi, Xj, i, j)
                        K[startj:endj, starti:endi] = K[starti:endi, startj:endj].T
                    
        elif itaskZ != None:
            '''
            Compute the block symmetric covariance matrix for different data points. 
            '''
            if len(itaskX) != len(itaskZ) and len(itaskX) != ntasks:
                raise ValueError('Insufficent number of task indices.')
            m = len(Z)
            itaskX = np.r_[itaskX, n]
            itaskZ = np.r_[itaskZ, m]
            
            print 'n={0},m={1}'.format(n,m)
            K = np.zeros((n,m))
            for i in xrange(ntasks):
                startiX = itaskX[i]
                endiX = itaskX[i+1]
                startiZ = itaskZ[i]
                endiZ = itaskZ[i+1]
                Xi = X[startiX:endiX,:]
                Zi = Z[startiZ:endiZ,:]
                K[startiX:endiX,startiZ:endiZ] = self.cov_block(Xi, Zi, i, i)
                for j in xrange(i+1,ntasks):
                    startjX = itaskX[j]
                    endjX = itaskX[j+1]
                    startjZ = itaskZ[j]
                    endjZ = itaskZ[j+1]
                    Xj = X[startjX:endjX,:]
                    Zj = Z[startjZ:endjZ,:] 
                    K[startiX:endiX,startjZ:endjZ] = self.cov_block(Xi, Zj, i, j)
                    K[startjX:endjX,startiZ:endiZ] = self.cov_block(Xj, Zi, j, i) #check iff works correctly
                    
            
        else:
            '''
            Compute the q-th column of the covariance matrix if q is specified, 
            otherwise we compute the cross cov matrix between all tasks of output X
            and latent points (inducing vars) Z.
            '''
            if len(itaskX) != ntasks:
                raise ValueError('Insufficent number of task indices.')
            if q != None and q > ntasks:
                raise ValueError('Unknown task: {0}'.format(q))
            
            m = len(Z)
            K = np.zeros((n,m))
            itaskX = np.r_[itaskX, n]
            for i in xrange(ntasks):
                    
                starti = itaskX[i]
                endi = itaskX[i+1]
                Xi = X[starti:endi,:]
                K[starti:endi,:] = self.cov_block(Xi, Z, i, q)            
        
        return K
        
        
    
    def cov_old(self, X, itask, Z=None, diag=False):
        '''
        Computes the full covariance for all tasks. If X just specified the returned
        matrix is of size DNxDN, otherwise if Z specified the size is DNxDM.
        '''
        ntasks = self._ntasks
        
        xeqz = Z == None
        
        n = X.shape[0]
        itask = np.r_[itask, n]
        if xeqz:
            m = n
            K = np.zeros((n,m)) if diag==False else np.zeros(n)
        else:
            m = Z.shape[0]
            K = np.zeros((n,m*ntasks))
        
        for i in xrange(ntasks):
            starti = itask[i]
            endi = itask[i+1]
            Xi = X[starti:endi,:]
    
            if xeqz:
                if diag:
                    K[starti:endi] = self.cov_block(Xi, i=i, diag=diag)
                else:
                    K[starti:endi,starti:endi] = self.cov_block(Xi, i=i, diag=diag)
                #if diag:
                    #K[starti:endi] = self.cov(Xi, i, dep=True, diag=True)
                    #K[starti:endi] += self.cov(Xi, i, dep=False, diag=True)
                #    
                #else:
                    #K[starti:endi, starti:endi] = self.cov(Xi, i, dep=True)
                    #K[starti:endi, starti:endi] += self.cov(Xi, i, dep=False)
            else: 
                K[starti:endi, i*m:(i+1)*m] = self.cov_block(Xi, Z, i, i)
                #K[starti:endi, i*m:(i+1)*m] = self.ccov(Xi, Z, i, i)
                #HACK: add the indeent part to the diag blocks of the matrix
                #idpKernel = self._idpKernel
                #if idpKernel != None:
                #    K[starti:endi, i*m:(i+1)*m] += idpKernel[i](Xi, Z)
                            
            if diag == False:
                for j in xrange(i+1,ntasks):
                    startj = itask[j]
                    endj = itask[j+1]
                    if xeqz:
                        Xj = X[startj:endj,:]
                        #K[starti:endi, startj:endj] = self.ccov(Xi, Xj, i, j)
                        K[starti:endi, startj:endj] = self.cov_block(Xi, Xj, i, j)
                        K[startj:endj, starti:endi] = K[starti:endi, startj:endj].T #TODO: check if the transpose is correct
                    else:
                        #K[starti:endi, j*m:(j+1)*m] = self.ccov(Xi, Z, i, j)
                        K[starti:endi, j*m:(j+1)*m] = self.cov_block(Xi, Z, i, j)
                        K[startj:endj, i*m:(i+1)*m] = K[starti:endi, j*m:(j+1)*m]
                    
                    
        return K
    
    def cov_block(self, X, Z=None, i=None, j=None, diag=False):
        '''
        '''
        xeqz = Z == None
        
        dpKernel = self._dpKernel
        idpKernel = self._idpKernel
        theta = self._theta
        
        if (xeqz and i == None) or (i == None and j == None):
            '''
            X and/or Z are not associated to a specific block(task), thus
            we compute only the cov matrix of the latent process, e.g. 
            computing cov matrix of the inducing points, need for sparse
            multioutput gps.
            '''
            K = dpKernel.lcov(X, Z, diag)
            
        elif xeqz or i==j:
            '''   
            Compute the blk diag part of the cov variance matrix. It is the
            auto covariance and includes the cov of the dependent and 
            independent kernel 
            '''
            thetaX = theta[i]
            if xeqz:
                K = dpKernel.cov(X, thetaX, diag)
                if idpKernel != None:
                    K += idpKernel[i](X, diag=diag) 
            else:
                K = dpKernel.ccov(X, Z, thetaX, thetaX) 
                if idpKernel != None:
                    K += idpKernel[i](X, Z)
                
        else:
            thetaX = theta[i]
            if j == None:
                '''
                Z is not associated to a specific block/task, thus we
                compute the cross cov matrix between output X (comes from task i)
                and latent points (inducing vars) Z.  
                '''
                K = dpKernel.ccov(X, Z, thetaX, latent=True)
            else:
                '''
                Compute the cross covariance of the block i and j between X and Z.
                '''
                thetaZ = theta[j]
                K = dpKernel.ccov(X, Z, thetaX, thetaZ, latent=False)
                
        return K
                
    def cov1(self, X, task=None, dep=True, diag=False):
        '''
        @deprecated: too complicated
        ''' 
        
        if task == None:
            '''
            No task is specified, we compute the latent covariance matrix.
            It is use when X are reflects a set of inducing points 
            '''
            dpKernel = self._dpKernel
            K = dpKernel.lcov(X, diag=diag)
        else:
            '''
            Computing the covariance matrix of X. We distinguish between
            the private kernel (dep==False) or the shared kernel (dep==True).
            '''
            if task > self._ntasks:
                    raise ValueError('unknown task: {0}'.format(task))

            if dep == True:
                theta = self._theta[task]
                dpKernel = self._dpKernel
                K = dpKernel.cov(X, theta, diag) 
            else:
                n = len(X)
                idpKernel = self._idpKernel[task]
                K = idpKernel(X, diag=diag) if idpKernel != None else (np.zeros((n,n)) if diag == False else np.zeros(n))
                
        return K
            
    def ccov(self, X, Z, taskX, taskZ=None, dep=True):
        '''
        @deprecated: too complicated 
        TODO: should we allow the access to the independent kernel
              + we need access to the covariance of the independent kernel for two different datasets,
                if we want to compute a sparse approximation. furthermore the noise model is often 
                encoded in the covariance function, so there is no way to exclude this feature, if we
                not allow the cov computation of two different sets.
              - per definition the cross covarianze of the independent kernel for two diffenent tasks
                is zero  
        '''        
        if taskX > self._ntasks:
            raise ValueError('unknown task for X: {0}'.format(taskX))
        if taskZ and taskZ > self._ntasks:
            raise ValueError('unknown task for Z: {0}'.format(taskZ))
        
        dpKernel = self._dpKernel
        thetaX = self._theta[taskX]
        thetaZ = self._theta[taskZ] if taskZ != None else None
        latent = (taskZ == None)
        
        K = dpKernel.ccov(X, Z, thetaX, thetaZ, latent)
        return K
    
    
    def gradient(self, covGrad, X, Z=None, itaskX=None, itaskZ=None, q=None, diag=False):
        '''
        '''
        '''
        '''
        xeqz = Z == None
        n = X.shape[0]
        
        dpKernel = self._dpKernel
        ntasks = self._ntasks
        
        nparams = self._nparams
        grad = np.zeros(nparams)
        
        if (xeqz and itaskX == None) or (itaskX == None and itaskZ == None):
            '''
            No block(task) indices are specified, therefore the covariance
            of the latent process is computed.
            '''
            #for ip in xrange(dpKernel.nparams):
            #    dKernel = dpKernel.derivate(ip)
            #    dK = dKernel.lcov(X, Z, diag)
            #    grad[ip] = np.sum(covGrad*dK)
        
                    
            offset = dpKernel.nparams
            grad[0:offset] = dpKernel.gradient_lcov(covGrad, X, Z, diag=diag)

            
            #K = dpKernel.lcov(X, Z, diag)
            
        elif xeqz:
            '''
            Compute full covariance matrix for X
            '''
            if len(itaskX) != ntasks:
                raise ValueError('Insufficent number of task indices.')
            
            itaskX = np.r_[itaskX, n]
            for i in xrange(ntasks):
                starti = itaskX[i]
                endi = itaskX[i+1]
                Xi = X[starti:endi,:]
                covGradBlk = covGrad[starti:endi] if diag else covGrad[starti:endi, starti:endi] 
                grad += self.gradient_block(covGradBlk, Xi, i=i, diag=diag)
                #print 'block:{0}{1}; gradient={2}'.format(i,i, grad)
                if diag==False:
                    for j in xrange(i+1, ntasks):
                        startj = itaskX[j]
                        endj = itaskX[j+1]
                        Xj = X[startj:endj,:]
                        covGradBlk = covGrad[starti:endi, startj:endj]
                        #@todo: check if the two times are correct
                        grad += 2*self.gradient_block(covGradBlk, Xi, Xj, i, j) 
                        #print 'block:{0}{1}; gradient={2}'.format(i,j, grad)
                
#            if diag:
#                K = np.zeros(n)
#                for i in xrange(ntasks):
#                    start = itaskX[i]
#                    end = itaskX[i+1]
#                    K[start:end] = self.cov_block(X[start:end,:], i=i, diag=True)
#            else: 
#                K = np.zeros((n,n))
#                for i in xrange(ntasks):
#                    starti = itaskX[i]
#                    endi = itaskX[i+1]
#                    Xi = X[starti:endi,:]
#                    K[starti:endi, starti:endi] = self.cov_block(Xi, i=i)
#                    for j in xrange(i+1, ntasks):
#                        startj = itaskX[j]
#                        endj = itaskX[j+1]
#                        Xj = X[startj:endj,:]
#                        K[starti:endi, startj:endj] = self.cov_block(Xi, Xj, i, j)
#                        K[startj:endj, starti:endi] = K[starti:endi, startj:endj].T
                    
        elif itaskZ != None:
            '''
            Compute the block symmetric covariance matrix for different data points. 
            '''
            if len(itaskX) != len(itaskZ) and len(itaskX) != ntasks:
                raise ValueError('Insufficent number of task indices.')
            m = len(Z)
            itaskX = np.r_[itaskX, n]
            itaskZ = np.r_[itaskZ, m]
            
            for i in xrange(ntasks):
                startiX = itaskX[i]
                endiX = itaskX[i+1]
                startiZ = itaskZ[i]
                endiZ = itaskZ[i+1]
                Xi = X[startiX:endiX,:]
                Zi = Z[startiZ:endiZ,:]
                covGradBlk = covGrad[startiX:endiX, startiZ:endiZ]
                grad += self.gradient_block(covGradBlk, Xi, Zi, i, i)
                #K[startiX:endiX,startiZ:endiZ] = self.cov_block(Xi, Zi, i, i)
                for j in xrange(i+1,ntasks):
                    #startjX = itaskX[j]
                    #endjX = itaskX[j+1]
                    startjZ = itaskZ[j]
                    endjZ = itaskZ[j+1]
                    Zj = Z[startjZ:endjZ,:]
                    covGradBlk = covGrad[startiX:endiX, startjZ:endjZ]
                    grad += 2*self.gradient_block(covGradBlk, Xi, Zj, i, j) #@todo: check if the two times are correct
                    #K[startiX:endiX,startjZ:endjZ] = self.cov_block(Xi, Zj, i, j)
                    #K[startjX:endjX,startiZ:endiZ] = K[startiX:endiX,startjZ:endjZ]
            
        else:
            '''
            Compute the q-th column of the covariance matrix if q is specified, 
            otherwise we compute the cross cov matrix between all tasks of output X
            and latent points (inducing vars) Z.
            '''
            if len(itaskX) != ntasks:
                raise ValueError('Insufficent number of task indices.')
            if q != None and q > ntasks:
                raise ValueError('Unknown task: {0}'.format(q))
            
            m = len(Z)
            itaskX = np.r_[itaskX, n]
            for i in xrange(ntasks):
                    starti = itaskX[i]
                    endi = itaskX[i+1]
                    Xi = X[starti:endi,:]
                    covGradBlk = covGrad[starti:endi]
                    grad += self.gradient_block(covGradBlk, Xi, Z, i, q)
                    #K[starti:endi,:] = self.cov_block(Xi, Z, i, q)            
        
        return grad

    def gradientX(self, covGrad, X, Z=None, itaskX=None, itaskZ=None, q=None, diag=False):
        '''
        '''
        '''
        '''
        xeqz = Z == None
        n = X.shape[0]
        
        dpKernel = self._dpKernel
        ntasks = self._ntasks
        
        #nparams = self._nparams
        #grad = np.zeros(nparams)

        m,d = X.shape
        gradX = np.zeros((m,d))
        dKernelX = dpKernel.derivateX()
        
        if (xeqz and itaskX == None) or (itaskX == None and itaskZ == None):
            '''
            No block(task) indices are specified, therefore the covariance
            of the latent process is computed.
            '''
            dX = dKernelX.lcov(X,Z,diag=diag)*2.0
            
            for i in xrange(m):
                for j in xrange(d):
                    gradX[i,j] = np.dot(dX[:,i,j], covGrad[:,i])
            
            
            
            #K = dpKernel.lcov(X, Z, diag)
            
        elif xeqz:
            '''
            Compute full covariance matrix for X
            '''
            if len(itaskX) != ntasks:
                raise ValueError('Insufficent number of task indices.')
            
            itaskX = np.r_[itaskX, n]
            for i in xrange(ntasks):
                starti = itaskX[i]
                endi = itaskX[i+1]
                Xi = X[starti:endi,:]
                covGradBlk = covGrad[starti:endi] if diag else covGrad[starti:endi, starti:endi] 
                gradX[starti:endi] = self.gradientX_block(covGradBlk, Xi, i=i, diag=diag)*2.0 #check if multiplying by 2 is correct, because of the symmetry?!             K[starti:endi, startj:endj] = self.cov_block(Xi, Xj, i, j)
#                        K[startj:endj, starti:endi] = K[starti:endi, startj:endj].T
                    
        elif itaskZ != None:
            '''
            Compute the block symmetric covariance matrix for different data points. 
            '''
            if len(itaskX) != len(itaskZ) and len(itaskX) != ntasks:
                raise ValueError('Insufficent number of task indices.')
            m = len(Z)
            itaskX = np.r_[itaskX, n]
            itaskZ = np.r_[itaskZ, m]
            
            for i in xrange(ntasks):
                startiX = itaskX[i]
                endiX = itaskX[i+1]
                startiZ = itaskZ[i]
                endiZ = itaskZ[i+1]
                Xi = X[startiX:endiX,:]
                Zi = Z[startiZ:endiZ,:]
                covGradBlk = covGrad[startiX:endiX, startiZ:endiZ]
                gradX += self.gradientX_block(covGradBlk, Xi, Zi, i, i)
                #K[startiX:endiX,startiZ:endiZ] = self.cov_block(Xi, Zi, i, i)
                for j in xrange(i+1,ntasks):
                    #startjX = itaskX[j]
                    #endjX = itaskX[j+1]
                    startjZ = itaskZ[j]
                    endjZ = itaskZ[j+1]
                    Zj = Z[startjZ:endjZ,:]
                    covGradBlk = covGrad[startiX:endiX, startjZ:endjZ]
                    gradX += 2*self.gradientX_block(covGradBlk, Xi, Zj, i, j) #@todo: check if the two times are correct
                    #K[startiX:endiX,startjZ:endjZ] = self.cov_block(Xi, Zj, i, j)
                    #K[startjX:endjX,startiZ:endiZ] = K[startiX:endiX,startjZ:endjZ]
            
        else:
            '''
            Compute the q-th column of the covariance matrix if q is specified, 
            otherwise we compute the cross cov matrix between all tasks of output X
            and latent points (inducing vars) Z.
            '''
            if len(itaskX) != ntasks:
                raise ValueError('Insufficent number of task indices.')
            if q != None and q > ntasks:
                raise ValueError('Unknown task: {0}'.format(q))
            
            m = len(Z)
            #Hack for derivate of latent points Z 
            if q == None:
                gradX = np.zeros((m,d))
            itaskX = np.r_[itaskX, n]
            for i in xrange(ntasks):
                    starti = itaskX[i]
                    endi = itaskX[i+1]
                    Xi = X[starti:endi,:]
                    covGradBlk = covGrad[starti:endi]
                    gradX += self.gradientX_block(covGradBlk, Xi, Z, i, q)
                    #K[starti:endi,:] = self.cov_block(Xi, Z, i, q)            
        
        return gradX
        
        
    def gradient_block(self, covGrad, X, Z=None, i=None, j=None, diag=False):
        '''
        '''
        xeqz = Z == None
        
        dpKernel = self._dpKernel
        idpKernel = self._idpKernel
        theta = self._theta
        
        ntasks = self._ntasks
        nparams = self._nparams
        grad = np.zeros(nparams)
        
        if (xeqz and i == None) or (i == None and j == None):
            '''
            X and/or Z are not associated to a specific block(task), thus
            we compute only the cov matrix of the latent process, e.g. 
            computing cov matrix of the inducing points, need for sparse
            multioutput gps.
            '''
            
            #for ip in xrange(dpKernel.nparams):
            #    dKernel = dpKernel.derivate(ip)
            #    dK = dKernel.lcov(X, Z, diag)
            #    grad[ip] = np.sum(covGrad*dK)
            
            offset = dpKernel.nparams
            grad[0:offset] = dpKernel.gradient_lcov(covGrad, X, Z, diag=diag)

            
        elif xeqz or i==j:
            '''   
            Compute the blk diag part of the cov variance matrix. It is the
            auto covariance and includes the cov of the dependent and 
            independent kernel 
            '''
            thetaX = theta[i]
            if xeqz:
                #for ip in xrange(dpKernel.nparams):
                #    dKernel = dpKernel.derivate(ip)
                #    dK = dKernel.cov(X, thetaX, diag=diag)
                #    grad[ip] = np.sum(covGrad*dK)
                
                #offset = dpKernel.nparams + i*dpKernel.ntheta
                #for ip in xrange(dpKernel.ntheta):
                #    dKernel = dpKernel.derivateTheta(ip)
                #    dK = dKernel.cov(X, thetaX, diag=diag)
                #    grad[offset+ip] = np.sum(covGrad*dK)
                    
                gradP, gradTheta = dpKernel.gradient_cov(covGrad, X, thetaX, diag=diag)
                offset = dpKernel.nparams
                grad[0:offset] = gradP
                offset = dpKernel.nparams + i*dpKernel.ntheta
                ntheta = dpKernel.ntheta
                grad[offset:offset+ntheta] = gradTheta
                
                if idpKernel != None:
                    offset = dpKernel.nparams + dpKernel.ntheta*ntasks + i*idpKernel[i].nparams
                    grad[offset:offset+idpKernel[i].nparams] = idpKernel[i].gradient(covGrad, X, diag=diag)
                    #for ip in xrange(idpKernel[i].nparams):
                    #    dKernel = idpKernel[i].derivate(ip)
                    #    dK = dKernel(X, diag=diag)
                    #    grad[offset+ip] = np.sum(covGrad*dK)
  
                #K = dpKernel.cov(X, thetaX, diag)
                #if idpKernel != None:
                #    K += idpKernel[i](X, diag=diag) 
            else:
                #for ip in xrange(dpKernel.nparams):
                #    dKernel = dpKernel.derivate(ip)
                #    dK = dKernel.ccov(X, Z, thetaX, thetaX)
                #    grad[ip] = np.sum(covGrad*dK)

                #offset = dpKernel.nparams + i*dpKernel.ntheta
                #for ip in xrange(dpKernel.ntheta):
                #    dKernel = dpKernel.derivateTheta(ip)
                #    dKx, dKz = dKernel.ccov(X, Z, thetaX, thetaX)
                #    grad[offset+ip] = np.sum(covGrad*dKx) + np.sum(covGrad*dKz) #Check if the correct

                gradP, gradThetaX, gradThetaZ = dpKernel.gradient_ccov(covGrad, X, Z, thetaX, thetaX)
                offset = dpKernel.nparams
                grad[0:offset] = gradP
                offset = dpKernel.nparams + i*dpKernel.ntheta
                ntheta = dpKernel.ntheta
                grad[offset:offset+ntheta] = gradThetaX+gradThetaZ

                
                if idpKernel != None:
                    offset = dpKernel.nparams + dpKernel.ntheta*ntasks + i*idpKernel[i].nparams
                    grad[offset:offset+idpKernel[i].nparams] = idpKernel[i].gradient(covGrad, X, Z)
                    #for ip in xrange(idpKernel[i].nparams):
                    #    dKernel = idpKernel[i].derivate(ip)
                    #    dK = dKernel(X, Z)
                    #    grad[offset+ip] = np.sum(covGrad*dK)
                
                #K = dpKernel.ccov(X, Z, thetaX, thetaX) 
                #if idpKernel != None:
                #    K += idpKernel[i](X, Z)
                
        else:
            thetaX = theta[i]
            offseti = dpKernel.nparams + i*dpKernel.ntheta
            if j == None:
                '''
                Z is not associated to a specific block/task, thus we
                compute the cross cov matrix between output X (comes from task i)
                and latent points (inducing vars) Z.  
                '''
                #for ip in xrange(dpKernel.nparams):
                #    dKernel = dpKernel.derivate(ip)
                #    dK = dKernel.ccov(X, Z, thetaX, latent=True)
                #    grad[ip] = np.sum(covGrad*dK)
                    
                #for ip in xrange(dpKernel.ntheta):
                #    dKernel = dpKernel.derivateTheta(ip)
                #    dK = dKernel.ccov(X, Z, thetaX, latent=True)
                #    grad[offseti+ip] = np.sum(covGrad*dK)
                
                gradP, gradTheta = dpKernel.gradient_ccov(covGrad, X, Z, thetaX, latent=True)
                grad[0:dpKernel.nparams] = gradP
                grad[offseti:offseti+dpKernel.ntheta] = gradTheta


                #K = dpKernel.ccov(X, Z, thetaX, latent=True)
            else:
                '''
                Compute the cross covariance of the block i and j between X and Z.
                '''
                thetaZ = theta[j]
                offsetj = dpKernel.nparams + j*dpKernel.ntheta
                #for ip in xrange(dpKernel.nparams):
                #    dKernel = dpKernel.derivate(ip)
                #    dK = dKernel.ccov(X, Z, thetaX, thetaZ, latent=False)
                #    grad[ip] = np.sum(covGrad*dK)
                    
                #for ip in xrange(dpKernel.ntheta):
                #    dKernel = dpKernel.derivateTheta(ip)
                #    dKx, dKz = dKernel.ccov(X, Z, thetaX, thetaX)
                #    grad[offseti+ip] = np.sum(covGrad*dKx) 
                #    grad[offsetj+ip] = np.sum(covGrad*dKz) 
                    
                #K = dpKernel.ccov(X, Z, thetaX, thetaZ, latent=False)
                
                gradP, gradThetaX, gradThetaZ = dpKernel.gradient_ccov(covGrad, X, Z, thetaX, thetaZ, latent=False)
                grad[0:dpKernel.nparams] = gradP
                grad[offseti:offseti+dpKernel.ntheta] = gradThetaX
                grad[offsetj:offsetj+dpKernel.ntheta] = gradThetaZ
                
        #print 'block:{0},{1}; gradient={2}'.format(i,j, grad)
        return grad

    def gradientX_block(self, covGrad, X, Z=None, i=None, j=None, diag=False):
        '''
        idpKernel are not include in gradient computation. if necessary, fix it.
        '''
        xeqz = Z == None
        
        dpKernel = self._dpKernel
        theta = self._theta
        
        m,d = X.shape
        dKernelX = dpKernel.derivateX()
        if (xeqz and i == None) or (i == None and j == None):
            '''
            X and/or Z are not associated to a specific block(task), thus
            we compute only the cov matrix of the latent process, e.g. 
            computing cov matrix of the inducing points, need for sparse
            multioutput gps.
            '''
            dX = dKernelX.lcov(X,Z,diag=diag)*2.0
            
        elif xeqz or i==j:
            '''   
            Compute the blk diag part of the cov variance matrix. It is the
            auto covariance and includes the cov of the dependent and 
            independent kernel 
            '''
            thetaX = theta[i]
            if xeqz:
                dX = dKernelX.cov(X, thetaX, diag=diag) 
            else:
                dX = dKernelX.cov(X, Z, thetaX, thetaX)
                                        
        else:
            thetaX = theta[i]
            if j == None:
                '''
                Z is not associated to a specific block/task, thus we
                compute the cross cov matrix between output X (comes from task i)
                and latent points (inducing vars) Z.  
                '''
                m = len(Z)
                #dX = dKernelX.ccov(X, Z, thetaX, latent=True)
                #hack: normally we are interested in the gradient of the inducing points z     
                dX = dKernelX.ccov(Z, X, thetaX, latent=True)    
                #K = dpKernel.ccov(X, Z, thetaX, latent=True)
            else:
                '''
                Compute the cross covariance of the block i and j between X and Z.
                '''
                thetaZ = theta[j]
                dX = dKernelX.ccov(X, Z, thetaX, thetaZ, latent=False)
        
        
        gradX = np.zeros((m,d))
        for i in xrange(m):
            for j in xrange(d):
                gradX[i,j] = np.dot(dX[:,i,j], covGrad[:,i])
                        
        #print 'block:{0},{1}; gradient={2}'.format(i,j, grad)
        return gradX


    def derivate(self, i):
        '''
        Returns the derivative function of the i-th parameter. The method
        is a relict of the base kernels. In a multioutput setting it can
        be tedious invoking this function to compute the gradient. The 
        reason for this lies in the fact on the block structure of the 
        cov matrix in which not all parts are influenced by all type of parameters.
        Thus the returned cov matrix is in some cases very sparse, especially
        by the parameters of the smoothing kernel (theta) and the parameters
        of the independent process. A further disposal of the derivate of the
        cov matrix can have high computational costs. So we suggest to make 
        fast computation use the gradient function, which is optimized internally.
        
        '''
    
    def derivateX(self):
        '''
        '''
    

    def derivate_block(self, i=None, dep=True):
        '''
        
        '''

        class DependentDerivativeFunc(object):
            '''
                Derivate of the latent kernel
            '''
            def __init__(self, kernel, task, theta=None):
                self.kernel = kernel #the latent kernel
                self.task = task
                self.theta = theta
                self.nparams = kernel.nparams if task == 0 else kernel.ntheta
                
            def cov(self, X, i, diag=False):
                '''
                @deprecated
                '''
                if i >= self.nparams:
                    raise ValueError('Unknown hyperparameter')
                
                task = self.task
                kernel = self.kernel
                
                if task == None:
                    #primary task
                    dKernel = kernel.derivate(i)
                    dK = dKernel.lcov(X, diag=diag)
                else:
                    dKernel = kernel.derivate(i) if i < kernel.params else kernel.derivateTheta(i-kernel.nparams)
                    theta = self.theta[task]
                    dK = dKernel.cov(X, theta, diag)
                return dK
                
            def ccov(self, X, Z, i, q=None):
                '''
                @deprecated
                TODO: throw an exception if the task is not specified
                '''
                if i >= self.nparams:
                    raise ValueError('Unknown hyperparameter')

                task = self.task
                kernel = self.kernel
                theta = self.theta[task]
                thetaQ = self.theta[q] if q else None
                latent = (q != None)
                dKernel = kernel.derivateTheta(i)
                return dKernel.ccov(X, Z, theta, thetaQ, latent) 
                
        class IndependentDerivateFunc(object):
            
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
            
            def ccov(self, X, Z, i):
                '''
                TODO: - ConvolvedMTL Kernel doesn't support the cross-covariance of the 
                        independent kernel. So these method should either throw an exception
                        or return an zero matrix
                '''
                if i >= self.nparams:
                    raise ValueError('Unknown hyperparameter')

                kernel = self.kernel
                dKernel = kernel.derivate(i)
                dK = dKernel(X, Z)
                return dK
        
        if dep:
            deriv_fct = DependentDerivativeFunc(self, task, self.theta)
        else:
            deriv_fct = IndependentDerivateFunc(self)
        
        return deriv_fct
        
    
    
    def _get_params(self):
        dpKernel = self._dpKernel
        idpKernel = self._idpKernel
        theta = self._theta
        
        ntasks = self._ntasks
        
        
        params = np.r_[dpKernel.params, theta.ravel()]
        if idpKernel != None:
            for i in xrange(ntasks):
                params = np.r_[params, idpKernel[i].params]
            
        return params
    
    def _set_params(self, params):
        dpKernel = self._dpKernel
        idpKernel = self._idpKernel
        
        ntasks = self._ntasks
        ntheta = dpKernel.ntheta
        
        dpKernel.params = params[0:dpKernel.nparams]
        offset = dpKernel.nparams
        self._theta = np.reshape(params[offset:offset+ntasks*ntheta], (ntasks, ntheta))
        offset += ntasks*ntheta
        if idpKernel != None:
            for i in xrange(ntasks):
                idpKernel[i].params = params[offset:offset+idpKernel[i].nparams]
                offset += idpKernel[i].nparams
        
    params = property(fget=_get_params, fset=_set_params)
    
    def _get_nparams(self):
        return self._nparams
    
    nparams = property(fget=_get_nparams)
    
    def dependent_kernel(self):
        return self._lKernel
    
    def independent_kernel(self, task):
        #kernel = None
        if task > self._ntasks:
            raise ValueError('unknown task: {0}'.format(task))
        kernel = self._idpKernel[task]
        return kernel

