'''
Created on May 22, 2012

@author: marcel
'''
import numpy as np

from upgeo.base.likel import LikelihoodFunction
from upgeo.amtl.infer import ApproxType
from scipy.linalg.decomp_cholesky import cho_solve

class PFITCFocusedGaussianLogLikel(LikelihoodFunction):
    '''
    @todo: - consider to define the input noise implicit in the likelihood function. 
             at the moment input noise can be formulated by the covariance function 
             as hyperparameter. 
    '''
    
    def __init__(self, gp_model):
        LikelihoodFunction.__init__(self, gp_model)
        
    def __call__(self):
        gp = self._gp
        
        Xp = gp._Xp
        Xu = gp._Xu
        Xs = gp._Xs
        yp = gp._yp
        ys = gp._ys
        y = np.r_[yp, ys]
        pKernel = gp._pKernel
        rhos = np.exp(gp._rhos)
        task_sizes = gp._task_sizes
        
        m = gp._m
        mp = gp._mp

        Ku = pKernel(Xu,Xu)
        Ksu = pKernel(Xs, Xu)
        Ksu = (Ksu.T * np.repeat(rhos, task_sizes)).T
        Kpu = pKernel(Xp, Xu)
        iKu = np.linalg.inv(Ku + np.eye(len(Xu))*1e-6)
        Kp = gp._Kp
        Ksp = np.dot(np.dot(Ksu, iKu), Kpu.T)
        
        Ks = np.dot(np.dot(Ksu, iKu), Ksu.T)
        #Kss = pKernel(Xs,Xs)
        G = np.diag(pKernel(Xs,Xs))*np.repeat(rhos**2,gp._task_sizes)-np.diag(Ks)
        
       
        Kpriv = gp._Kpriv
        K = np.zeros((m,m))
        K[0:mp, 0:mp] = Kp
        K[mp:m, 0:mp] = Ksp
        K[0:mp, mp:m] = Ksp.T
        
        K[mp:m, mp:m] = Ks + np.diag(G) + Kpriv
        
        L = np.linalg.cholesky(K)
        
        r = np.linalg.solve(L, y)
        
        
        likel = (-np.dot(r, r) - m*np.log(2.0*np.pi)) / 2.0
        likel -= np.sum(np.log(np.diag(L)))
        #likel -= np.log(np.linalg.det(Kp))/2.0
        
        return likel
    
    def gradient(self):
        gp = self._gp
        
        #@todo look for a better way to store the proximity

        rhos = np.exp(gp._rhos)
        Xu = gp._Xu
        Xp = gp._Xp
        Xs = gp._Xs
        yp = gp._yp
        ys = gp._ys
        y = np.r_[yp, ys]
        task_sizes = gp._task_sizes
        
        
        m = gp._m
        mp = gp._mp
        mtp = gp._mtp
        mts = gp._mts
        mtasks = gp._mtasks
        pKernel = gp._pKernel

        Ku = pKernel(Xu,Xu)
        Ksu = pKernel(Xs, Xu)
        Ksu = (Ksu.T * np.repeat(rhos, task_sizes)).T
        Kpu = pKernel(Xp, Xu)
        iKu = np.linalg.inv(Ku + np.eye(len(Xu))*1e-6)
        Kp = gp._Kp
        Ksp = np.dot(np.dot(Ksu, iKu), Kpu.T)
        
        Ks = np.dot(np.dot(Ksu, iKu), Ksu.T)
        #Kss = pKernel(Xs,Xs)
        G = np.diag(pKernel(Xs,Xs))*np.repeat(rhos**2,gp._task_sizes)-np.diag(Ks)
        
       
        Kpriv = gp._Kpriv
        K = np.zeros((m,m))
        K[0:mp, 0:mp] = Kp
        K[mp:m, 0:mp] = Ksp
        K[0:mp, mp:m] = Ksp.T
        
        K[mp:m, mp:m] = Ks + np.diag(G) + Kpriv
        
        #K = np.eye(m)*1e-18
        L = np.linalg.cholesky(K)
        iK = cho_solve((L,1), np.eye(m))
        alpha = np.dot(iK, y)
        
        
        #debugging
        #K1 = np.empty((m,m))
        #K[0:mp, 0:mp] = gp._pKernel(gp._Xp)
        #K[mp:m, 0:mp] = gp._sKernel(gp._Xs, gp._Xp)
        #params = gp.hyperparams
        #priors = gp._priors
        #print 'grad'
        Q = np.outer(alpha, alpha)-iK
        #Q = np.outer(alpha, alpha)

        dtheta_p = np.zeros(mtp)
        dtheta_s = np.zeros((mtasks, mts))
        drho = np.zeros(mtasks)
        
        #determine the gradient of the kernel hyperparameters
        #gradK = np.empty(k)
        for i in xrange(mtp):
            #compute the gradient of the i-th hyperparameter
            dK = self._gradient_thetap(i)
            dtheta_p[i] = np.sum(Q*dK) / 2.0
                    
        for i in xrange(mtasks):
            for j in xrange(mts):
                dK = self._gradient_thetas(i, j)
                dtheta_s[i,j] = np.sum(Q*dK) / 2.0

        for i in xrange(mtasks):
            #compute the gradient of the i-th hyperparameter
            dK = self._gradient_rhos(i)
            drho[i] = np.sum(Q*dK) / 2.0
            
        drho = drho*rhos
        grad = np.hstack((dtheta_p, dtheta_s.ravel(), drho))
        #print 'grad time={0}'.format(time.time()-t)
        return grad   
    
    def _gradient_thetap(self, i):
        gp = self._gp
        
        m = gp._m
        mp = gp._mp
        Xp = gp._Xp
        Xu = gp._Xu
        Xs = gp._Xs
        
        rhos = np.exp(gp._rhos)
        task_sizes = gp._task_sizes
        
        pKernel = gp._pKernel
        dpKernel = pKernel.derivate(i)
        
        
        
        dKp = dpKernel(Xp)
        
        Ku = pKernel(Xu,Xu)
        Ksu = pKernel(Xs, Xu)
        Ksu = (Ksu.T * np.repeat(rhos, task_sizes)).T
        Kpu = pKernel(Xp, Xu)
        iKu = np.linalg.inv(Ku + np.eye(len(Xu))*1e-6)
        #Ksp = np.dot(np.dot(Ksu, iKu), Kpu.T)

        dKu = dpKernel(Xu, Xu)
        dKsu = dpKernel(Xs, Xu)
        dKsu = (dKsu.T * np.repeat(rhos, task_sizes)).T
        dKpu = dpKernel(Xp, Xu)
        
        dKsp = np.dot(np.dot(dKsu, iKu), Kpu.T) + np.dot(np.dot(Ksu, iKu), dKpu.T) - np.dot(np.dot(np.dot(np.dot(Ksu, iKu), dKu), iKu), Kpu.T)
        
        #Ks = np.dot(np.dot(Ksu, iKu), Ksu.T)
        dKs = np.dot(np.dot(dKsu, iKu), Ksu.T) + np.dot(np.dot(Ksu, iKu), dKsu.T) - np.dot(np.dot(np.dot(np.dot(Ksu, iKu), dKu), iKu), Ksu.T)  
        #Kss = pKernel(Xs,Xs)
        dG = np.diag(dpKernel(Xs,Xs))*np.repeat(rhos**2,gp._task_sizes)-np.diag(dKs)
        
        
        #dKs = dKq+dG
        dK = np.zeros((m,m))
        dK[0:mp, 0:mp] = dKp
        dK[mp:m, 0:mp] = dKsp
        dK[0:mp, mp:m] = dKsp.T
        dK[mp:m, mp:m] = dKs + np.diag(dG)
        return dK

    def _gradient_thetas(self, task, i):
        gp = self._gp
        
        m = gp._m
        mp = gp._mp
        ms = gp._ms
        Xs = gp._Xs
        task_sizes = gp._task_sizes
        itask = np.r_[gp._itask, ms]
        
        sKernel = gp._sKernels[task]
        dsKernel = sKernel.derivate(i)
        
        start = itask[task]
        end = itask[task+1]
        dKpriv = dsKernel(Xs[start:end])
        
        
         
        dK = np.zeros((m,m))
        #dK[0:mp, 0:mp] = np.zeros(mp, mp)
        #dK[mp:m, 0:mp] = np.zeros(ms, mp)
        #dK[0:mp, mp:m] = np.zeros(mp, ms)
        dK[mp+start:mp+end, mp+start:mp+end] = dKpriv
        return dK        

    def _gradient_rhos(self, i):
        gp = self._gp
        
        task_sizes = gp._task_sizes
        m = gp._m
        mp = gp._mp
        ms = gp._ms
        Xu = gp._Xu
        Xp = gp._Xp
        Xs = gp._Xs
        rhos = np.exp(gp._rhos)
        #task_sizes = gp._task_sizes
        itask = np.r_[gp._itask, ms]
        pKernel = gp._pKernel
        
        start = itask[i]
        end = itask[i+1]
        
        Ku = pKernel(Xu,Xu)
        Ksu = pKernel(Xs, Xu)
        Kpu = pKernel(Xp, Xu)
        iKu = np.linalg.inv(Ku + np.eye(len(Xu))*1e-6)
        #Ksp = np.dot(np.dot(Ksu, iKu), Kpu.T)

        dKsu = np.zeros((ms, len(Xu)))
        dKsu[start:end] = Ksu[start:end] 
        Ksu = (Ksu.T * np.repeat(rhos, task_sizes)).T
        
        #dKsu = dpKernel(Xs, Xu)
        #dKsu = (dKsu.T * np.repeat(rhos, task_sizes)).T
        #dKpu = dpKernel(Xp, Xu)
        dKsp = np.dot(np.dot(dKsu, iKu), Kpu.T) 
        
        #Ks = np.dot(np.dot(Ksu, iKu), Ksu.T)
        dKs = np.dot(np.dot(dKsu, iKu), Ksu.T) + np.dot(np.dot(Ksu, iKu), dKsu.T)  
        #Kss = pKernel(Xs,Xs)
        #dG = np.zeros(ms)
        dKss = np.zeros(ms)
        dKss[start:end] = 2.0*np.diag(pKernel(Xs[start:end],Xs[start:end]))*rhos[i]
        dG = dKss-np.diag(dKs)

        dK = np.zeros((m,m))
        #dK[0:mp, 0:mp] = dKp
       
        dK[mp:, 0:mp] = dKsp
        dK[0:mp, mp:] = dKsp.T
        #dK[mp+start:mp+end, mp+start:mp+end] = dKq
        dK[mp:m, mp:m] = dKs + np.diag(dG)
        #dKs[:,start:end] = dKs[:,start:end] + np.diag(dG)
        #dK[mp+start:mp+end, mp:] = dKs 
        return dK


class SimpleFocusedGaussianLogLikel(LikelihoodFunction):
    '''
    @todo: - consider to define the input noise implicit in the likelihood function. 
             at the moment input noise can be formulated by the covariance function 
             as hyperparameter. 
    '''
    
    def __init__(self, gp_model):
        LikelihoodFunction.__init__(self, gp_model)
        
    def __call__(self):
        gp = self._gp
        
        Xp = gp._Xp
        yp = gp._yp
        ys = gp._ys
        y = np.r_[yp, ys]
        
        m = gp._m
        mp = gp._mp


        Kp = gp._Kp
        Ksp = gp._Ksp
        Lq = gp._Lq
       
        nfKp = gp._pKernel(Xp, Xp)
        nfiKp = np.linalg.inv(nfKp + np.eye(mp)*1e-6)
       
        Kpriv = gp._Kpriv
        G = gp._G
        K = np.zeros((m,m))
        K[0:mp, 0:mp] = Kp
        K[mp:m, 0:mp] = Ksp
        K[0:mp, mp:m] = Ksp.T
        K[mp:m, mp:m] = np.dot(Lq.T, Lq) + np.diag(G) + Kpriv
        #K[mp:m, mp:m] = np.dot(Lq.T, Lq) + Kpriv
        #K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfiKp), Ksp.T) + Kpriv
        #K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfiKp), Ksp.T) + np.diag(G) + Kpriv
        #K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfiKp), Ksp.T)  + Kpriv
        #K[mp:m, mp:m] = np.dot(Ksp,Ksp.T)+Kpriv
        #K = np.eye(m)*1e-18
        L = np.linalg.cholesky(K)
        
        #debugging
        #K1 = np.empty((m,m))
        #K[0:mp, 0:mp] = gp._pKernel(gp._Xp)
        #K[mp:m, 0:mp] = gp._sKernel(gp._Xs, gp._Xp)
        #params = gp.hyperparams
        #priors = gp._priors
        r = np.linalg.solve(L, y)
        
        
        #print 'complexity term={0}'.format(-np.sum(np.log(np.diag(L))))
        #print 'complexity term1={0}'.format(-np.log(np.linalg.det(K))/2)
        #print 'data fit term={0}'.format(-np.dot(r, r)/2.0)
        #print 'norm fit term={0}'.format(-m*np.log(2.0*np.pi)/2.0)
        
        likel = (-np.dot(r, r) - m*np.log(2.0*np.pi)) / 2.0
        likel -= np.sum(np.log(np.diag(L)))
        #likel -= np.log(np.linalg.det(Kp))/2.0
        
        #if priors != None:
        #    for i in xrange(len(params)):
        #        pr = priors[i]
        #        if pr != None:
        #            likel += pr.log_pdf(params[i])
        
        return likel
    
    def gradient(self):
        gp = self._gp
        
        #@todo look for a better way to store the proximity

        rhos = np.exp(gp._rhos)
        Xp = gp._Xp
        yp = gp._yp
        ys = gp._ys
        y = np.r_[yp, ys]
        
        m = gp._m
        mp = gp._mp
        mtp = gp._mtp
        mts = gp._mts
        mtasks = gp._mtasks

        Kp = gp._Kp
        Ksp = gp._Ksp
        Lq = gp._Lq

        nfKp = gp._pKernel(Xp, Xp)
        nfiKp = np.linalg.inv(nfKp + np.eye(mp)*1e-6)

       
        Kpriv = gp._Kpriv
        G = gp._G
        K = np.zeros((m,m))
        K[0:mp, 0:mp] = Kp
        K[mp:m, 0:mp] = Ksp
        K[0:mp, mp:m] = Ksp.T
        K[mp:m, mp:m] = np.dot(Lq.T, Lq) + np.diag(G) + Kpriv
        #K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfiKp), Ksp.T) + np.diag(G) + Kpriv
        #K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfiKp), Ksp.T)  + Kpriv
        #K[mp:m, mp:m] = np.dot(Lq.T, Lq) + Kpriv
        #K[mp:m, mp:m] = np.dot(Ksp,Ksp.T)+Kpriv
        
        #K = np.eye(m)*1e-18
        L = np.linalg.cholesky(K)
        iK = cho_solve((L,1), np.eye(m))
        alpha = np.dot(iK, y)
        
        
        #debugging
        #K1 = np.empty((m,m))
        #K[0:mp, 0:mp] = gp._pKernel(gp._Xp)
        #K[mp:m, 0:mp] = gp._sKernel(gp._Xs, gp._Xp)
        #params = gp.hyperparams
        #priors = gp._priors
        #print 'grad'
        Q = np.outer(alpha, alpha)-iK
        #Q = np.outer(alpha, alpha)

        dtheta_p = np.zeros(mtp)
        dtheta_s = np.zeros((mtasks, mts))
        drho = np.zeros(mtasks)
        
        #determine the gradient of the kernel hyperparameters
        #gradK = np.empty(k)
        for i in xrange(mtp):
            #compute the gradient of the i-th hyperparameter
            dK = self._gradient_thetap(i)
            dtheta_p[i] = np.sum(Q*dK) / 2.0
                    
        for i in xrange(mtasks):
            for j in xrange(mts):
                dK = self._gradient_thetas(i, j)
                dtheta_s[i,j] = np.sum(Q*dK) / 2.0

        for i in xrange(mtasks):
            #compute the gradient of the i-th hyperparameter
            dK = self._gradient_rhos(i)
            drho[i] = np.sum(Q*dK) / 2.0
            
        drho = drho*rhos
        grad = np.hstack((dtheta_p, dtheta_s.ravel(), drho))
        #print 'grad time={0}'.format(time.time()-t)
        return grad   
    
    def _gradient_thetap(self, i):
        gp = self._gp
        
        m = gp._m
        mp = gp._mp
        ms = gp._ms
        Xp = gp._Xp
        Xs = gp._Xs
        Ksp = gp._Ksp
        rhos = np.exp(gp._rhos)
        task_sizes = gp._task_sizes
        itask = np.r_[gp._itask, ms]
        pKernel = gp._pKernel
        dpKernel = pKernel.derivate(i)
        nfKp = pKernel(Xp, Xp)
        nfiKp = np.linalg.inv(nfKp + np.eye(mp)*1e-6)
        
        dKp = dpKernel(Xp)
        nfdKp = dpKernel(Xp,Xp)
        dKsp = dpKernel(Xs,Xp)
        dKsp = (dKsp.T*rhos.repeat(task_sizes)).T
        dKq = np.dot(np.dot(dKsp, nfiKp), Ksp.T) + np.dot(np.dot(Ksp, nfiKp), dKsp.T) - np.dot(np.dot(np.dot(np.dot(Ksp, nfiKp), nfdKp), nfiKp), Ksp.T)
        ddiagKs = np.diag(dpKernel(Xs,Xs))*np.repeat(rhos**2.0, task_sizes)
        dG = ddiagKs - np.diag(dKq)
        dKs = dKq+np.diag(dG)
        #dKs = dKq+dG
        dK = np.zeros((m,m))
        dK[0:mp, 0:mp] = dKp
        dK[mp:m, 0:mp] = dKsp
        dK[0:mp, mp:m] = dKsp.T
        dK[mp:m, mp:m] = dKs
        return dK

    def _gradient_thetas(self, task, i):
        gp = self._gp
        
        m = gp._m
        mp = gp._mp
        ms = gp._ms
        Xs = gp._Xs
        task_sizes = gp._task_sizes
        itask = np.r_[gp._itask, ms]
        
        sKernel = gp._sKernels[task]
        dsKernel = sKernel.derivate(i)
        
        start = itask[task]
        end = itask[task+1]
        dKpriv = dsKernel(Xs[start:end])
        
        
         
        dK = np.zeros((m,m))
        #dK[0:mp, 0:mp] = np.zeros(mp, mp)
        #dK[mp:m, 0:mp] = np.zeros(ms, mp)
        #dK[0:mp, mp:m] = np.zeros(mp, ms)
        dK[mp+start:mp+end, mp+start:mp+end] = dKpriv
        return dK        

    def _gradient_rhos(self, i):
        gp = self._gp
        
        task_sizes = gp._task_sizes
        m = gp._m
        mp = gp._mp
        ms = gp._ms
        Xp = gp._Xp
        Xs = gp._Xs
        Ksp = gp._Ksp
        rhos = np.exp(gp._rhos)
        #task_sizes = gp._task_sizes
        itask = np.r_[gp._itask, ms]
        pKernel = gp._pKernel
        #dpKernel = pKernel.derivate(i)
        nfKp = pKernel(Xp, Xp)
        nfiKp = np.linalg.inv(nfKp + np.eye(mp)*1e-6)
        #nfiKp = gp._iKp
        diagKs = gp._diagKs
        
        start = itask[i]
        end = itask[i+1]
        dKsp = np.zeros((ms,mp))
        ddiagKs = np.zeros(ms)    
        #dKsp = (Ksp.T/rhos.repeat(task_sizes)).T
        
        dKsp[start:end] = Ksp[start:end,:]/rhos[i]
        ddiagKs[start:end] = 2.0*diagKs[start:end] / rhos[i]
        dKq = np.dot(np.dot(dKsp, nfiKp), Ksp.T) + np.dot(np.dot(Ksp, nfiKp), dKsp.T)
        #dKq = np.dot(dKsp, Ksp.T)+np.dot(Ksp,dKsp.T)
        dG = ddiagKs - np.diag(dKq)
        dKs = dKq+np.diag(dG)
               
        dK = np.zeros((m,m))
        #dK[0:mp, 0:mp] = dKp
        dK[mp+start:mp+end, 0:mp] = dKsp[start:end]
        dK[0:mp, mp+start:mp+end] = dKsp[start:end].T
        #dK[mp+start:mp+end, mp+start:mp+end] = dKq
        dK[mp:m, mp:m] = dKs
        return dK


class FTCFocusedGaussianLogLikel(LikelihoodFunction):
    '''
    @todo: - consider to define the input noise implicit in the likelihood function. 
             at the moment input noise can be formulated by the covariance function 
             as hyperparameter. 
    '''
    
    def __init__(self, gp_model):
        LikelihoodFunction.__init__(self, gp_model)
        
    def __call__(self):
        gp = self._gp
        
        Xp = gp._Xp
        Xs = gp._Xs
        yp = gp._yp
        ys = gp._ys
        y = np.r_[yp, ys]
        
        m = gp._m
        mp = gp._mp


        rhos = np.exp(gp._rhos)
        Kp = gp._Kp
        Ksp = gp._Ksp
        Lq = gp._Lq
        Ks = gp._pKernel(Xs,Xs)
        Ks = Ks * np.repeat(rhos,gp._task_sizes)
        Ks = (Ks.T * np.repeat(rhos,gp._task_sizes)).T
       
        nfKp = gp._pKernel(Xp, Xp)
        nfiKp = np.linalg.inv(nfKp + np.eye(mp)*1e-6)
       
        Kpriv = gp._Kpriv
        G = gp._G
        K = np.zeros((m,m))
        K[0:mp, 0:mp] = Kp
        K[mp:m, 0:mp] = Ksp
        K[0:mp, mp:m] = Ksp.T
        K[mp:m, mp:m] = Ks + Kpriv
        #K[mp:m, mp:m] = np.dot(Lq.T, Lq) + Kpriv
        #K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfiKp), Ksp.T) + Kpriv
        #K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfiKp), Ksp.T) + np.diag(G) + Kpriv
        #K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfiKp), Ksp.T)  + Kpriv
        #K[mp:m, mp:m] = np.dot(Ksp,Ksp.T)+Kpriv
        #K = np.eye(m)*1e-18
        L = np.linalg.cholesky(K)
        
        #debugging
        #K1 = np.empty((m,m))
        #K[0:mp, 0:mp] = gp._pKernel(gp._Xp)
        #K[mp:m, 0:mp] = gp._sKernel(gp._Xs, gp._Xp)
        #params = gp.hyperparams
        #priors = gp._priors
        r = np.linalg.solve(L, y)
        
        
        #print 'complexity term={0}'.format(-np.sum(np.log(np.diag(L))))
        #print 'complexity term1={0}'.format(-np.log(np.linalg.det(K))/2)
        #print 'data fit term={0}'.format(-np.dot(r, r)/2.0)
        #print 'norm fit term={0}'.format(-m*np.log(2.0*np.pi)/2.0)
        
        likel = (-np.dot(r, r) - m*np.log(2.0*np.pi)) / 2.0
        likel -= np.sum(np.log(np.diag(L)))
        #likel -= np.log(np.linalg.det(Kp))/2.0
        
        #if priors != None:
        #    for i in xrange(len(params)):
        #        pr = priors[i]
        #        if pr != None:
        #            likel += pr.log_pdf(params[i])
        
        return likel
    
    def gradient(self):
        gp = self._gp
        
        #@todo look for a better way to store the proximity

        rhos = np.exp(gp._rhos)
        Xp = gp._Xp
        Xs = gp._Xs
        yp = gp._yp
        ys = gp._ys
        y = np.r_[yp, ys]
        
        m = gp._m
        mp = gp._mp
        mtp = gp._mtp
        mts = gp._mts
        mtasks = gp._mtasks

        Kp = gp._Kp
        Ksp = gp._Ksp
        Lq = gp._Lq

        nfKp = gp._pKernel(Xp, Xp)
        nfiKp = np.linalg.inv(nfKp + np.eye(mp)*1e-6)

        rhos = np.exp(gp._rhos)
        Kp = gp._Kp
        Ksp = gp._Ksp
        Lq = gp._Lq
        Ks = gp._pKernel(Xs,Xs)
        Ks = Ks * np.repeat(rhos,gp._task_sizes)
        Ks = (Ks.T * np.repeat(rhos,gp._task_sizes)).T
       
        nfKp = gp._pKernel(Xp, Xp)
        nfiKp = np.linalg.inv(nfKp + np.eye(mp)*1e-6)
       
        Kpriv = gp._Kpriv
        G = gp._G
        K = np.zeros((m,m))
        K[0:mp, 0:mp] = Kp
        K[mp:m, 0:mp] = Ksp
        K[0:mp, mp:m] = Ksp.T
        K[mp:m, mp:m] = Ks + Kpriv
        #K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfiKp), Ksp.T) + np.diag(G) + Kpriv
        #K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfiKp), Ksp.T)  + Kpriv
        #K[mp:m, mp:m] = np.dot(Lq.T, Lq) + Kpriv
        #K[mp:m, mp:m] = np.dot(Ksp,Ksp.T)+Kpriv
        
        #K = np.eye(m)*1e-18
        L = np.linalg.cholesky(K)
        iK = cho_solve((L,1), np.eye(m))
        alpha = np.dot(iK, y)
        
        
        #debugging
        #K1 = np.empty((m,m))
        #K[0:mp, 0:mp] = gp._pKernel(gp._Xp)
        #K[mp:m, 0:mp] = gp._sKernel(gp._Xs, gp._Xp)
        #params = gp.hyperparams
        #priors = gp._priors
        #print 'grad'
        Q = np.outer(alpha, alpha)-iK
        #Q = np.outer(alpha, alpha)

        dtheta_p = np.zeros(mtp)
        dtheta_s = np.zeros((mtasks, mts))
        drho = np.zeros(mtasks)
        
        #determine the gradient of the kernel hyperparameters
        #gradK = np.empty(k)
        for i in xrange(mtp):
            #compute the gradient of the i-th hyperparameter
            dK = self._gradient_thetap(i)
            dtheta_p[i] = np.sum(Q*dK) / 2.0
                    
        for i in xrange(mtasks):
            for j in xrange(mts):
                dK = self._gradient_thetas(i, j)
                dtheta_s[i,j] = np.sum(Q*dK) / 2.0

        for i in xrange(mtasks):
            #compute the gradient of the i-th hyperparameter
            dK = self._gradient_rhos(i)
            drho[i] = np.sum(Q*dK) / 2.0
            
        drho = drho*rhos
        grad = np.hstack((dtheta_p, dtheta_s.ravel(), drho))
        #print 'grad time={0}'.format(time.time()-t)
        return grad   
    
    def _gradient_thetap(self, i):
        gp = self._gp
        
        m = gp._m
        mp = gp._mp
        ms = gp._ms
        Xp = gp._Xp
        Xs = gp._Xs
        Ksp = gp._Ksp
        rhos = np.exp(gp._rhos)
        task_sizes = gp._task_sizes
        itask = np.r_[gp._itask, ms]
        pKernel = gp._pKernel
        dpKernel = pKernel.derivate(i)
        
        dKp = dpKernel(Xp)
        dKsp = dpKernel(Xs,Xp)
        dKsp = (dKsp.T*rhos.repeat(task_sizes)).T
        
        
        dKs = dpKernel(Xs,Xs)
        dKs = dKs*np.repeat(rhos, task_sizes)
        dKs = (dKs.T*np.repeat(rhos, task_sizes))
        
        #dKs = dKq+dG
        dK = np.zeros((m,m))
        dK[0:mp, 0:mp] = dKp
        dK[mp:m, 0:mp] = dKsp
        dK[0:mp, mp:m] = dKsp.T
        dK[mp:m, mp:m] = dKs
        return dK

    def _gradient_thetas(self, task, i):
        gp = self._gp
        
        m = gp._m
        mp = gp._mp
        ms = gp._ms
        Xs = gp._Xs
        task_sizes = gp._task_sizes
        itask = np.r_[gp._itask, ms]
        
        sKernel = gp._sKernels[task]
        dsKernel = sKernel.derivate(i)
        
        start = itask[task]
        end = itask[task+1]
        dKpriv = dsKernel(Xs[start:end])
        
        
         
        dK = np.zeros((m,m))
        #dK[0:mp, 0:mp] = np.zeros(mp, mp)
        #dK[mp:m, 0:mp] = np.zeros(ms, mp)
        #dK[0:mp, mp:m] = np.zeros(mp, ms)
        dK[mp+start:mp+end, mp+start:mp+end] = dKpriv
        return dK        

    def _gradient_rhos(self, i):
        gp = self._gp
        
        task_sizes = gp._task_sizes
        m = gp._m
        mp = gp._mp
        ms = gp._ms
        Xp = gp._Xp
        Xs = gp._Xs
        Ksp = gp._Ksp
        rhos = np.exp(gp._rhos)
        #task_sizes = gp._task_sizes
        itask = np.r_[gp._itask, ms]
        pKernel = gp._pKernel
        #dpKernel = pKernel.derivate(i)
        nfKp = pKernel(Xp, Xp)
        nfiKp = np.linalg.inv(nfKp + np.eye(mp)*1e-6)
        #nfiKp = gp._iKp
        diagKs = gp._diagKs
        
        start = itask[i]
        end = itask[i+1]
        dKsp = np.zeros((ms,mp))
        ddiagKs = np.zeros(ms)
        dKsp[start:end] = Ksp[start:end,:]/rhos[i]    
        #dKsp = (Ksp.T/rhos.repeat(task_sizes)).T
        
        Ks = np.zeros((ms,ms))
        Ks[start:end,:] = pKernel(Xs[start:end],Xs)
        dKs = Ks * rhos.repeat(task_sizes)
        dKs[start:end,start:end] *= 2
        dKs[:start,start:end] = dKs[start:end,:start].T
        dKs[end:,start:end] = dKs[start:end,end:].T
        
        dK = np.zeros((m,m))
        #dK[0:mp, 0:mp] = dKp
        dK[mp+start:mp+end, 0:mp] = dKsp[start:end]
        dK[0:mp, mp+start:mp+end] = dKsp[start:end].T
        #dK[mp+start:mp+end, mp+start:mp+end] = dKq
        dK[mp:m, mp:m] = dKs
        return dK
 
 
class PITCFocusedGaussianLogLikel(LikelihoodFunction):
    '''
    @todo: - consider to define the input noise implicit in the likelihood function. 
             at the moment input noise can be formulated by the covariance function 
             as hyperparameter. 
    '''
    
    def __init__(self, gp_model):
        LikelihoodFunction.__init__(self, gp_model)
        
    def __call__(self):
        gp = self._gp
        

        Xs = gp._Xs
        yp = gp._yp
        ys = gp._ys
        y = np.r_[yp, ys]
        
        m = gp._m
        mp = gp._mp
        ms = gp._ms

        Kp = gp._Kp
        Ksp = gp._Ksp
        Lq = gp._Lq
        G = gp._G
        Kpriv = gp._Kpriv
        
        mtasks = gp._mtasks
        itask = np.r_[gp._itask, ms]
        task_sizes = gp._task_sizes
        rhos = np.exp(gp._rhos)
        Ks = np.zeros((ms,ms))
        #G = np.zeros((ms,ms))
        #Q = np.dot(Lq.T, Lq)
        #for i in xrange(mtasks):
        #    start = itask[i]
        #    end = itask[i+1]
        #    Ks[start:end,start:end] = gp._pKernel(Xs[start:end]) * rhos[i]**2
        #    G[start:end,start:end] = Ks[start:end,start:end]-Q[start:end,start:end]  
       
        K = np.empty((m,m))
        K[0:mp, 0:mp] = Kp
        K[mp:m, 0:mp] = Ksp
        K[0:mp, mp:m] = Ksp.T
        K[mp:m, mp:m] = np.dot(Lq.T, Lq)  + G + Kpriv
        #K[mp:m, mp:m] = Ks + Kpriv
        #K[mp:m, mp:m] = Ks + Kpriv
        #print 'Kpriv'
        #print Kpriv
        #print 'Kpriv oarams'
        #print gp._sKernels[0].params
        L = np.linalg.cholesky(K)
        
        
        nfKp = gp._pKernel(gp._Xp, gp._Xp)
        nfiKp = np.linalg.inv(nfKp + np.eye(mp)*1e-6)

        #print 'gug'
        #print K[mp:m, mp:m]
        #print np.dot(np.dot(Ksp, nfiKp), Ksp.T) - np.dot(Lq.T, Lq)
        #debugging
        #K1 = np.empty((m,m))
        #K[0:mp, 0:mp] = gp._pKernel(gp._Xp)
        #K[mp:m, 0:mp] = gp._sKernel(gp._Xs, gp._Xp)
        #params = gp.hyperparams
        #priors = gp._priors
        r = np.linalg.solve(L, y)
        
        #print 'complexity term={0}'.format(-np.sum(np.log(np.diag(L))))
        #print 'complexity term1={0}'.format(-np.log(np.linalg.det(K))/2)
        #print 'data fit term={0}'.format(-np.dot(r, r)/2.0)
        #print 'norm fit term={0}'.format(-m*np.log(2.0*np.pi)/2.0)
        
        likel = (-np.dot(r, r) - m*np.log(2.0*np.pi)) / 2.0
        likel -= np.sum(np.log(np.diag(L)))
        #likel -= np.log(np.linalg.det(Kp))/2.0
        
        #if priors != None:
        #    for i in xrange(len(params)):
        #        pr = priors[i]
        #        if pr != None:
        #            likel += pr.log_pdf(params[i])
        
        return likel
    
    def gradient(self):
        gp = self._gp
        
        #@todo look for a better way to store the proximity

        rhos = np.exp(gp._rhos)
        Xp = gp._Xp
        yp = gp._yp
        ys = gp._ys
        y = np.r_[yp, ys]
        
        m = gp._m
        mp = gp._mp
        mtp = gp._mtp
        mts = gp._mts
        mtasks = gp._mtasks

        Kp = gp._Kp
        Ksp = gp._Ksp
        Lq = gp._Lq

        nfKp = gp._pKernel(gp._Xp, gp._Xp)
        nfiKp = np.linalg.inv(nfKp + np.eye(mp)*1e-6)

        #print 'gug'
        #print np.dot(np.dot(Ksp, nfiKp), Ksp.T) - np.dot(Lq.T, Lq)
       
        Kpriv = gp._Kpriv
        G = gp._G
        K = np.zeros((m,m))
        K[0:mp, 0:mp] = Kp
        K[mp:m, 0:mp] = Ksp
        K[0:mp, mp:m] = Ksp.T
        K[mp:m, mp:m] = np.dot(Lq.T, Lq) + G + Kpriv
        #K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfiKp), Ksp.T) + np.diag(G) + Kpriv
        #K[mp:m, mp:m] = np.dot(np.dot(Ksp, nfiKp), Ksp.T)  + Kpriv
        #K[mp:m, mp:m] = np.dot(Lq.T, Lq) + Kpriv
        #K[mp:m, mp:m] = np.dot(Ksp,Ksp.T)+Kpriv
        
        #K = np.eye(m)*1e-18
        L = np.linalg.cholesky(K)
        iK = cho_solve((L,1), np.eye(m))
        alpha = np.dot(iK, y)
        
        
        #debugging
        #K1 = np.empty((m,m))
        #K[0:mp, 0:mp] = gp._pKernel(gp._Xp)
        #K[mp:m, 0:mp] = gp._sKernel(gp._Xs, gp._Xp)
        #params = gp.hyperparams
        #priors = gp._priors
        #print 'grad'
        Q = np.outer(alpha, alpha)-iK
        #Q = np.outer(alpha, alpha)

        dtheta_p = np.zeros(mtp)
        dtheta_s = np.zeros((mtasks, mts))
        drho = np.zeros(mtasks)
        
        #determine the gradient of the kernel hyperparameters
        #gradK = np.empty(k)
        for i in xrange(mtp):
            #compute the gradient of the i-th hyperparameter
            dK = self._gradient_thetap(i)
            dtheta_p[i] = np.sum(Q*dK) / 2.0
                    
        for i in xrange(mtasks):
            for j in xrange(mts):
                dK = self._gradient_thetas(i, j)
                dtheta_s[i,j] = np.sum(Q*dK) / 2.0

        for i in xrange(mtasks):
            #compute the gradient of the i-th hyperparameter
            dK = self._gradient_rhos(i)
            drho[i] = np.sum(Q*dK) / 2.0
            
        drho = drho*rhos
        grad = np.hstack((dtheta_p, dtheta_s.ravel(), drho))
        #print 'grad time={0}'.format(time.time()-t)
        return grad   
    
    def _gradient_thetap(self, i):
        gp = self._gp
        
        m = gp._m
        mp = gp._mp
        ms = gp._ms
        Xp = gp._Xp
        Xs = gp._Xs
        Ksp = gp._Ksp
        rhos = np.exp(gp._rhos)
        task_sizes = gp._task_sizes
        itask = np.r_[gp._itask, ms]
        mtasks = gp._mtasks
        pKernel = gp._pKernel
        dpKernel = pKernel.derivate(i)
        nfKp = pKernel(Xp, Xp)
        nfiKp = np.linalg.inv(nfKp + np.eye(mp)*1e-6)
        
        dKp = dpKernel(Xp)
        nfdKp = dpKernel(Xp,Xp)
        dKsp = dpKernel(Xs,Xp)
        dKsp = (dKsp.T*rhos.repeat(task_sizes)).T
        
        dKq = np.dot(np.dot(dKsp, nfiKp), Ksp.T) + np.dot(np.dot(Ksp, nfiKp), dKsp.T) - np.dot(np.dot(np.dot(np.dot(Ksp, nfiKp), nfdKp), nfiKp), Ksp.T)
        dG = np.zeros((ms,ms))
        for j in xrange(mtasks):
            start = itask[j]
            end = itask[j+1]
            ddiagKs = dpKernel(Xs[start:end],Xs[start:end]) * rhos[j]**2 #noise free alternative
            #dG = ddiagKs - np.dot(R[:,start:end].T, iKq[:,start:end])
            #todo:optimize
            #dG = ddiagKs - (np.dot(dKsp[start:end,:], np.dot(iKp, Ksp[start:end,:].T)) + np.dot(np.dot(Ksp[start:end,:],iKp), dKsp[start:end,:].T) -np.dot(iKq[:,start:end].T, np.dot(dKp, iKq[:,start:end])))
            dG[start:end,start:end] = ddiagKs - dKq[start:end,start:end]
        
        dKs = dKq+dG
        #dKs = dKq+dG
        dK = np.zeros((m,m))
        dK[0:mp, 0:mp] = dKp
        dK[mp:m, 0:mp] = dKsp
        dK[0:mp, mp:m] = dKsp.T
        dK[mp:m, mp:m] = dKs
        return dK

    def _gradient_thetas(self, task, i):
        gp = self._gp
        
        m = gp._m
        mp = gp._mp
        ms = gp._ms
        Xs = gp._Xs
        task_sizes = gp._task_sizes
        itask = np.r_[gp._itask, ms]
        
        sKernel = gp._sKernels[task]
        dsKernel = sKernel.derivate(i)
        
        start = itask[task]
        end = itask[task+1]
        dKpriv = dsKernel(Xs[start:end])
        
        
         
        dK = np.zeros((m,m))
        #dK[0:mp, 0:mp] = np.zeros(mp, mp)
        #dK[mp:m, 0:mp] = np.zeros(ms, mp)
        #dK[0:mp, mp:m] = np.zeros(mp, ms)
        dK[mp+start:mp+end, mp+start:mp+end] = dKpriv
        return dK        

    def _gradient_rhos(self, i):
        gp = self._gp
        
        task_sizes = gp._task_sizes
        m = gp._m
        mp = gp._mp
        ms = gp._ms
        Xp = gp._Xp
        Xs = gp._Xs
        Ksp = gp._Ksp
        rhos = np.exp(gp._rhos)
        #task_sizes = gp._task_sizes
        itask = np.r_[gp._itask, ms]
        mtasks = gp._mtasks
        pKernel = gp._pKernel
        #dpKernel = pKernel.derivate(i)
        nfKp = pKernel(Xp, Xp)
        nfiKp = np.linalg.inv(nfKp + np.eye(mp)*1e-6)
        #nfiKp = gp._iKp
        diagKs = gp._diagKs
        
        start = itask[i]
        end = itask[i+1]
        dKsp = np.zeros((ms,mp))
        ddiagKs = np.zeros((ms,ms))    
        #dKsp = (Ksp.T/rhos.repeat(task_sizes)).T
        
        dKsp[start:end] = Ksp[start:end,:]/rhos[i]
        ddiagKs[start:end,start:end] = 2.0*diagKs[start:end,start:end] / rhos[i]
        dKq = np.dot(np.dot(dKsp, nfiKp), Ksp.T) + np.dot(np.dot(Ksp, nfiKp), dKsp.T)
        dG = np.zeros((ms,ms))
        for j in xrange(mtasks):
            jstart = itask[j]
            jend = itask[j+1]
            #dKq = np.dot(dKsp, Ksp.T)+np.dot(Ksp,dKsp.T)
            dG[jstart:jend,jstart:jend] = ddiagKs[jstart:jend,jstart:jend] - dKq[jstart:jend,jstart:jend]
        dKs = dKq+dG
        
        dK = np.zeros((m,m))
        #dK[0:mp, 0:mp] = dKp
        dK[mp+start:mp+end, 0:mp] = dKsp[start:end]
        dK[0:mp, mp+start:mp+end] = dKsp[start:end].T
        #dK[mp+start:mp+end, mp+start:mp+end] = dKq
        dK[mp:m, mp:m] = dKs
        return dK

    
class FocusedGaussianLogLikel(LikelihoodFunction):
    '''
    @todo: - consider to define the input noise implicit in the likelihood function. 
             at the moment input noise can be formulated by the covariance function 
             as hyperparameter. 
    '''
    
    def __init__(self, gp_model):
        LikelihoodFunction.__init__(self, gp_model)
            
    def __call__(self):
        gp = self._gp
        
        ys = gp._ys
        
        m = gp._m
        ms = gp._ms
        mtasks = gp._mtasks
        itask = np.r_[gp._itask, ms] #add the total number of sec tasks for easier iteration
        
        Ld = gp._Ld
        Lp = gp._Lp
        Lpy = gp._Lpy
        Ksp = gp._Ksp
        
        rp = gp._rp
        rsp = gp._rsp
        rs = gp._rs
        
        #noise free stuff
        nfiKp = np.linalg.inv(gp._pKernel(gp._Xp,gp._Xp)+np.diag(np.ones(gp._mp)*1e-8))
        nfrp = np.dot(nfiKp, gp._yp)

               
        Lqpy = np.zeros(ms)         #yp*iKp*Kps*Ld^(-1)
        Lsy = np.zeros(ms)          #Ld^(-1)*ys
        for i in xrange(mtasks):
            start = itask[i]
            end = itask[i+1]
            Lqpy[start:end] = np.linalg.solve(Ld[start:end,start:end], rsp[start:end])
            Lsy[start:end] = np.linalg.solve(Ld[start:end, start:end], ys[start:end])
            
        L1 = np.dot(Lpy, Lpy)+np.dot(Lqpy, Lqpy)
        #L2 = -2.0*np.dot(np.dot(rp, Ksp.T), rs)
        L2 = -2.0*np.dot(np.dot(nfrp, Ksp.T), rs) #noise free
        L3 = np.dot(Lsy, Lsy)
        
        likel = 0#(-L1 - L2 - L3 - m*np.log(2.0*np.pi))/2.0
        #likel = (- L2- m*np.log(2.0*np.pi))/2.0
        likel -= np.sum(np.log(np.diag(Lp))) + np.sum(np.log(np.diag(Ld)))
        
        #if priors != None:
        #    for i in xrange(len(params)):
        #        pr = priors[i]
        #        if pr != None:
        #            likel += pr.log_pdf(params[i])
        
        #print 'rho={0}'.format(gp._rhos)
        #print 'likel={0}'.format(likel)
        #print 'likel time={0}'.format(time.time()-t)
        
        return likel
    
    def gradient(self):
        approx = self._gp._infer.approx_type
        
        gp = self._gp

        Xp = gp._Xp        
        Xs = gp._Xs
        
        Ksp = gp._Ksp
        iD = gp._iD
        iKp = gp._iKp
        diagKs = gp._diagKs
        rp = gp._rp
        rs = gp._rs
        rdsp = gp._rdsp
        rips = gp._rips
        
        pKernel = gp._pKernel
        sKernels = gp._sKernels
        rhos = np.exp(gp._rhos) #rhos in log space->transform to linear space
        
        mtp = gp._mtp #number of hyperparameters for the primary kernel
        mts = gp._mts #number of hyperparameters for the secondary kernel
        
        ms = gp._ms
        mtasks = gp._mtasks
        itask = np.r_[gp._itask, ms] #add the total number of sec tasks for easier iteration
        task_sizes = gp._task_sizes


        dtheta_p = np.zeros(mtp)
        dtheta_s = np.zeros((mtasks, mts))
        drho = np.zeros(mtasks)
        
        #t1 = time.time()
        #iKq = np.dot(iKp, Ksp.T) #cho_solve((Lp, 1), Ksp.T)
        #noise free stuff
        iKq = np.dot(np.linalg.inv(pKernel(Xp,Xp)+np.diag(np.ones(gp._mp)*1e-16)), Ksp.T) #noise-free alternative
        nfiKp = np.linalg.inv(pKernel(Xp,Xp)+np.diag(np.ones(gp._mp)*1e-16)) #noise free
        nfrp = np.dot(nfiKp, gp._yp)

        #print 'grad proc time={0}'.format(time.time()-t1)
    
        #gradients for the hyperparameters of the primary kernel wrt. to the log likelihood
        #t1 = time.time()
        for i in xrange(mtp):
            dpKernel = pKernel.derivate(i)
            dKp = dpKernel(Xp)
            dKsp = dpKernel(Xs, Xp)
            dKsp = (dKsp.T*rhos.repeat(task_sizes)).T #multiply by the correlation parameter
            #R = 2.0*dKsp.T - np.dot(dKp, iKq) #dot product has high computationally if rank(Kp) is high
            nfdKp = dpKernel(Xp,Xp)
            R = 2.0*dKsp.T - np.dot(nfdKp, iKq) #noise free alternative
            
            ddetKp = np.sum(iKp*dKp) #d|Kp|/dtheta_p = trace(Kp^(-1)*dKp/dtheta_p)
            
            if approx == ApproxType.FITC:
                #ddiagKs = dpKernel(Xs, diag=True) * np.repeat(rhos**2.0, task_sizes) #multiply by the correlation parameter
                ddiagKs = np.diag(dpKernel(Xs, Xs)) * np.repeat(rhos**2.0, task_sizes) #noise freea alternative
                dG = ddiagKs - np.sum(R*iKq, 0)
                
                ddetD = np.dot(np.diag(iD),dG)     #d|D|/dtheta_p = trace(D^(-1)*dG/dtheta_p)
                drdspG = rdsp*dG
                drsG = rs*dG

            elif approx == ApproxType.PITC:
                drdspG = np.zeros(ms)
                drsG = np.zeros(ms)
            
                ddetD = 0                          #d|D|/dtheta_p = trace(D^(-1)*dG/dtheta_p)
                for j in xrange(mtasks):
                    start = itask[j]
                    end = itask[j+1]
                    #todo: check if correct
                    #ddiagKs = dpKernel(Xs[start:end]) * rhos[j]**2
                    ddiagKs = dpKernel(Xs[start:end],Xs[start:end]) * rhos[j]**2 #noise free alternative
                    
                    #dG = ddiagKs - np.dot(R[:,start:end].T, iKq[:,start:end])
                    #todo:optimize
                    #dG = ddiagKs - (np.dot(dKsp[start:end,:], np.dot(iKp, Ksp[start:end,:].T)) + np.dot(np.dot(Ksp[start:end,:],iKp), dKsp[start:end,:].T) -np.dot(iKq[:,start:end].T, np.dot(dKp, iKq[:,start:end])))
                    dG = ddiagKs - (np.dot(dKsp[start:end,:], np.dot(nfiKp, Ksp[start:end,:].T)) + np.dot(np.dot(Ksp[start:end,:],nfiKp), dKsp[start:end,:].T) -np.dot(iKq[:,start:end].T, np.dot(nfdKp, iKq[:,start:end])))
                    ddetD += np.sum(iD[start:end,start:end]*dG)
                    drdspG[start:end] = np.dot(rdsp[start:end],dG)
                    drsG[start:end] = np.dot(rs[start:end],dG)
            else:
                raise ValueError('Unknown approx type: {0}'.format(approx))
            
            
            #gradient of dtheta_p wrt L1
            dL1a = np.dot(np.dot(rp, dKp), rp)
            #dL1b = 2*np.dot(rp, np.dot(dKp, np.dot(iKp, np.dot(Ksp.T, rdsp))))
            dL1b = 2*np.dot(nfrp, np.dot(nfdKp, np.dot(nfiKp, np.dot(Ksp.T, rdsp)))) #noise free
            #dL1b -= 2*np.dot(np.dot(rp, dKsp.T), rdsp)
            dL1b -= 2*np.dot(np.dot(nfrp, dKsp.T), rdsp) #noise free
            dL1b += np.dot(drdspG,rdsp)
            #gradient of dtheta_p wrt L2
            #dL2 = -np.dot(np.dot(rp, dKp), rips) 
            dL2 = -np.dot(np.dot(nfrp, nfdKp), rips)#noise free
            #dL2 += np.dot(np.dot(rp, dKsp.T), rs)
            dL2 += np.dot(np.dot(nfrp, dKsp.T), rs)#noise free
            dL2 -= np.dot(drdspG, rs)
            dL2 *= 2.0
            
            #gradient of dtheta_p wrt L3
            dL3 = np.dot(drsG, rs)
            
                    
            dtheta_p[i] = -(ddetD+ddetKp)/2.0 + (dL1a + dL1b + dL2 + dL3)/2.0
            #dtheta_p[i] = -(ddetD)/2.0 #+ (dL1a+dL1b)/2.0
            #dtheta_p[i] = (dL2)/2.0
            
            
        #print 'grad theta_p time={0}'.format(time.time()-t1)
        #t1 = time.time()    
        for i in xrange(mtasks):
            start = itask[i]
            end = itask[i+1]
            for j in xrange(mts):
                dsKernel = sKernels[i].derivate(j) #most time comsuming part
                dKpriv = dsKernel(Xs[start:end])
                
                ddetD = np.sum(iD[start:end, start:end]*dKpriv)
                
                #gradient of drho wrt L1
                dL1b = np.dot(np.dot(rdsp[start:end], dKpriv), rdsp[start:end])
                
                #gradient of drho wrt L2
                dL2 = -np.dot(np.dot(rdsp[start:end], dKpriv), rs[start:end])
                dL2 *= 2.0
                #gradient of drho wrt L3
                dL3 = np.dot(np.dot(rs[start:end], dKpriv), rs[start:end])
                dtheta_s[i,j] = (-ddetD+dL1b+dL2+dL3)/2.0
                #dtheta_s[i,j] = (-ddetD)/2.0
        
        #print 'grad thdeta_s time={0}'.format(time.time()-t1)
        #t1 = time.time()
        #gradients for the correlation parameters wrt. to the log likelihood
        for i in xrange(mtasks):
            start = itask[i]
            end = itask[i+1]
            
            dKsp = Ksp[start:end,:] / rhos[i]
            if approx == ApproxType.FITC:
                ddiagKs = 2.0*diagKs[start:end] / rhos[i]
                dG = ddiagKs - 2.0 * np.sum(dKsp.T*iKq[:,start:end],0)
                
                ddetD = np.dot(np.diag(iD[start:end, start:end]),dG)
                
                drdspG = rdsp[start:end]*dG
                drsG = rs[start:end]*dG
            elif approx == ApproxType.PITC:
                ddiagKs = 2.0*diagKs[start:end,start:end] / rhos[i]
                dG = ddiagKs - 2.0 * np.dot(dKsp, iKq[:,start:end])
                
                ddetD = np.sum(iD[start:end,start:end]*dG)
                
                drdspG = np.dot(rdsp[start:end],dG)
                drsG = np.dot(rs[start:end],dG)
            else:
                raise ValueError('Unknown approx type: {0}'.format(approx))
            
            
            #gradient of drho wrt L1
            #dL1b = -2.0*np.dot(np.dot(rp, dKsp.T), rdsp[start:end])
            dL1b = -2.0*np.dot(np.dot(nfrp, dKsp.T), rdsp[start:end]) #noise free
            dL1b += np.dot(drdspG, rdsp[start:end])
            
            #gradient of drho wrt L2
            #dL2 = np.dot(np.dot(rp, dKsp.T), rs[start:end])
            dL2 = np.dot(np.dot(nfrp, dKsp.T), rs[start:end]) #noise free
            dL2 -= np.dot(drdspG, rs[start:end])
            dL2 *= 2.0
            
            #gradient of drho wrt L3
            dL3 = np.dot(drsG, rs[start:end])
            
            drho[i] = (-ddetD+dL1b+dL2+dL3) / 2.0    
            #drho[i] = (-ddetD) / 2.0
            
        drho = drho*rhos
        #print 'grad_rho={0}'.format(drho)
        #print 'grad rho time={0}'.format(time.time()-t1)
        grad = np.hstack((dtheta_p, dtheta_s.ravel(), drho))
        #print 'grad time={0}'.format(time.time()-t)
        return grad
    
class FMOGPGaussianLogLikel(LikelihoodFunction):
    '''
    @todo: - consider to define the input noise implicit in the likelihood function. 
             at the moment input noise can be formulated by the covariance function 
             as hyperparameter. 
    '''
    
    def __init__(self, gp_model):
        LikelihoodFunction.__init__(self, gp_model)
        
    def __call__(self):
        import time
        t = time.time()
        gp = self._gp
        
        ys = gp._ys
        
        m = gp._m
        ms = gp._ms
        mtasks = gp._mtasks
        itask = np.r_[gp._itask, ms] #add the total number of sec tasks for easier iteration
        
        Ld = gp._Ld
        Lp = gp._Lp
        Lpy = gp._Lpy
        Ksp = gp._Ksp
        
        rp = gp._rp
        rsp = gp._rsp
        rs = gp._rs
               
        Lqpy = np.zeros(ms)         #yp*iKp*Kps*Ld^(-1)
        Lsy = np.zeros(ms)          #Ld^(-1)*ys
        for i in xrange(mtasks):
            start = itask[i]
            end = itask[i+1]
            Lqpy[start:end] = np.linalg.solve(Ld[start:end,start:end], rsp[start:end])
            Lsy[start:end] = np.linalg.solve(Ld[start:end, start:end], ys[start:end])
            
        L1 = np.dot(Lpy, Lpy)+np.dot(Lqpy, Lqpy)
        L2 = -2.0*np.dot(np.dot(rp, Ksp.T), rs)
        L3 = np.dot(Lsy, Lsy)
        
        #likel = 0
        likel = (-L1 - L2 - L3 - m*np.log(2.0*np.pi))/2.0
        #likel = (-L1 - m*np.log(2.0*np.pi))/2.0
        likel -= np.sum(np.log(np.diag(Lp))) + np.sum(np.log(np.diag(Ld)))
        #likel -=  np.sum(np.log(np.diag(Lp)))
        
        #if priors != None:
        #    for i in xrange(len(params)):
        #        pr = priors[i]
        #        if pr != None:
        #            likel += pr.log_pdf(params[i])
        
        #print 'likel={0}'.format(likel)
        #print 'likel time={0}'.format(time.time()-t)
        
        return likel
    
    def gradient(self):
        import time
        t = time.time()
        
        gp = self._gp

        Xp = gp._Xp        
        Xs = gp._Xs
        
        Ksp = gp._Ksp
        iD = gp._iD
        iKp = gp._iKp
        rp = gp._rp
        rs = gp._rs
        rdsp = gp._rdsp
        rips = gp._rips
        
        kernel = gp._kernel
        
        mp = gp._mp
        ms = gp._ms
        mtasks = gp._mtasks
        itask = np.r_[gp._itask, ms] #add the total number of sec tasks for easier iteration
        
        #t1 = time.time()
        iKq = np.dot(iKp, Ksp.T) #cho_solve((Lp, 1), Ksp.T)
        #print 'grad proc time={0}'.format(time.time()-t1)
    
        #gradients for the hyperparameters of the primary kernel wrt. to the log likelihood
        #t1 = time.time()
        
        #1) compute the gradients for the parameters of the latent kernel associated to the primary task 
        dpKernel = kernel.derivate(task=0, latent=True) #derivative of the primary kernel
        dtheta_p = np.zeros(dpKernel.nparams)
        for i in xrange(dpKernel.nparams):
            dKp = dpKernel.cov(Xp, i)
            dKsp = np.zeros((ms,mp))
            ddiagKs = np.zeros(ms) #problems to compute this kernel gradient
            for j in xrange(mtasks):
                start = itask[j]
                end = itask[j+1]
                dKsp[start:end,:] = dpKernel.cross_cov(Xs[start:end], Xp, i, j+1)
                ddiagKs[start:end] = dpKernel.cov(Xs[start:end], i, j+1, diag=True)
            
            
            R = 2.0*dKsp.T - np.dot(dKp, iKq) #dot product has high computationally if rank(Kp) is high
            dG = ddiagKs - np.sum(R*iKq, 0)
            
            ddetD = np.dot(np.diag(iD),dG)     #d|D|/dtheta_p = trace(D^(-1)*dG/dtheta_p)
            ddetKp = np.sum(iKp*dKp)           #d|Kp|/dtheta_p = trace(Kp^(-1)*dKp/dtheta_p)

            dL1a = np.dot(np.dot(rp, dKp), rp)
            dL1b = 2*np.dot(rp, np.dot(dKp, np.dot(iKp, np.dot(Ksp.T, rdsp))))
            dL1b -= 2*np.dot(np.dot(rp, dKsp.T), rdsp)
            dL1b += np.dot(rdsp*dG,rdsp)

            dL2 = -np.dot(np.dot(rp, dKp), rips) 
            dL2 += np.dot(np.dot(rp, dKsp.T), rs)
            dL2 -= np.dot(rdsp*dG, rs)
            dL2 *= 2.0

            dL3 = np.dot(rs*dG, rs)        
            dtheta_p[i] = -(ddetD+ddetKp)/2.0 + (dL1a + dL1b + dL2 + dL3)/2.0
            #dtheta_p[i] = -(ddetKp)/2.0 + (dL1a + dL1b)/2 

        #2) compute the gradients for the parameters of the smoothing kernels of the secondary tasks
        dtheta_s = np.empty(0)
        for j in xrange(mtasks):
            dsKernel = kernel.derivate(task=j+1, latent=True)
            start = itask[j]
            end = itask[j+1]
            dtheta_sj = np.zeros(dsKernel.nparams)
            for i in xrange(dsKernel.nparams):
                dKsp = dsKernel.cross_cov(Xs[start:end], Xp, i)
                ddiagKs = dsKernel.cov(Xs[start:end], i, diag=True)
                dG = ddiagKs - 2.0 * np.sum(dKsp.T*iKq[:,start:end],0)
                ddetD = np.dot(np.diag(iD[start:end, start:end]),dG)

                #gradient of drho wrt L1
                dL1b = -2.0*np.dot(np.dot(rp, dKsp.T), rdsp[start:end])
                dL1b += np.dot(rdsp[start:end]*dG, rdsp[start:end])
                
                #gradient of drho wrt L2
                dL2 = np.dot(np.dot(rp, dKsp.T), rs[start:end])
                dL2 -= np.dot(rdsp[start:end]*dG, rs[start:end])
                dL2 *= 2.0
                
                #gradient of drho wrt L3
                dL3 = np.dot(rs[start:end]*dG, rs[start:end])
                #TODO: look for a geeignete data structure to store the gradients
                dtheta_sj[i] = (-ddetD+dL1b+dL2+dL3) / 2.0
            
            dtheta_s = np.r_[dtheta_s, dtheta_sj]
                
        #3) compute the gradients for the parameters of the independent kernel of the primary task
        dpKernel = kernel.derivate(task=0, latent=False)
        dtheta_ip = np.zeros(dpKernel.nparams)
        for i in xrange(dpKernel.nparams):
            dKp = dpKernel.cov(Xp, i)
            
            R = np.dot(dKp, iKq) #dot product has high computationally if rank(Kp) is high
            dG = np.sum(R*iKq, 0)
           
            ddetD = np.dot(np.diag(iD),dG)     #d|D|/dtheta_p = trace(D^(-1)*dG/dtheta_p)
            ddetKp = np.sum(iKp*dKp)           #d|Kp|/dtheta_p = trace(Kp^(-1)*dKp/dtheta_p)

            dL1a = np.dot(np.dot(rp, dKp), rp)
            dL1b = 2*np.dot(rp, np.dot(dKp, np.dot(iKp, np.dot(Ksp.T, rdsp))))
            dL1b += np.dot(rdsp*dG,rdsp)

            dL2 = -np.dot(np.dot(rp, dKp), rips) 
            dL2 -= np.dot(rdsp*dG, rs)
            dL2 *= 2.0

            dL3 = np.dot(rs*dG, rs)        
            dtheta_ip[i] = -(ddetD+ddetKp)/2.0 + (dL1a + dL1b + dL2 + dL3)/2.0
            #dtheta_ip[i] = (dL1a + dL1b + dL2 + dL3)/2.0
            #dtheta_ip[i] = -(ddetKp)/2.0 + (dL1a + dL1b + dL2 + dL3)/2.0
        
        #4) compute the gradients for the parameters of the indepent kernels of the secondary tasks
        dtheta_is = np.empty(0)     
        for j in xrange(mtasks):
            start = itask[j]
            end = itask[j+1]
            dsKernel = kernel.derivate(task=j+1, latent=False)
            dtheta_isj = np.zeros(dsKernel.nparams)
            for i in xrange(dsKernel.nparams):
                dKpriv = dsKernel.cov(Xs[start:end], i)
                ddetD = np.sum(iD[start:end, start:end]*dKpriv)
                
                #gradient of drho wrt L1
                dL1b = np.dot(np.dot(rdsp[start:end], dKpriv), rdsp[start:end])
                
                #gradient of drho wrt L2
                dL2 = -np.dot(np.dot(rdsp[start:end], dKpriv), rs[start:end])
                dL2 *= 2.0
                #gradient of drho wrt L3
                dL3 = np.dot(np.dot(rs[start:end], dKpriv), rs[start:end])
                dtheta_isj[i] = (-ddetD+dL1b+dL2+dL3)/2.0
                
            dtheta_is = np.r_[dtheta_is, dtheta_isj]
                
                
        #print 'grad rho time={0}'.format(time.time()-t1)
        print 'grad_theta_p={0}'.format(dtheta_p)
        print 'grad_theta_s={0}'.format(dtheta_s)
        print 'grad_theta_ip={0}'.format(dtheta_ip)
        print 'grad_theta_is={0}'.format(dtheta_is)
        grad = np.hstack((dtheta_p, dtheta_s, dtheta_ip, dtheta_is))
        print 'grad time={0}'.format(time.time()-t)
        return grad    