'''
Created on May 22, 2012

@author: marcel
'''
import numpy as np

from scipy.linalg.decomp_cholesky import cho_solve
from upgeo.base.kernel import SEKernel, NoiseKernel
    
if __name__ == '__main__':
    

    kernel = SEKernel(np.log(2), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(1)*np.ones(3), np.log(5)) #+ NoiseKernel(np.log(0.5))
    
    Xp = np.array([[0.2433, 0.3, 0.1283],
                   [0.1245, 0.563, 0.234],
                   [0.7454, -0.874, -0.85]])
    
    Xs =  np.array( [[-0.5046,    0.3999,   -0.5607],
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
    
    Kp = kernel(Xp)
    Ksp = kernel(Xs, Xp)
    Ks = kernel(Xs, diag=True)
    
    Lp = np.linalg.cholesky(Kp)
    
    print np.dot(np.dot(Ksp, np.linalg.inv(Kp)), Ksp.T)
    print np.dot(Ksp, cho_solve((Lp,1), Ksp.T))
    Lsp = np.linalg.solve(Lp, Ksp.T)
    print np.dot(Lsp.T, Lsp)
    
    
    G = Ks - np.sum(Lsp*Lsp, 0)
    Qs = np.dot(Lsp.T, Lsp) + np.diag(G)
    print Qs
    
    print 'Lsp={0}'.format(Lsp)
    
    print '-----------------'
    A = Kp + np.dot(np.dot(Ksp.T, np.linalg.inv(np.diag(G))), Ksp)
    iA = np.linalg.inv(A)
    S = np.dot(np.dot(Ksp, iA), Ksp.T)
    
    
    r = np.dot(np.linalg.inv(np.diag(G)), y)
    print np.dot(np.dot(r, S),r)
    
    V = np.linalg.solve(Lp, Ksp.T)
    G = Ks-np.sum(V*V, 0)
    print 'V={0}'.format(V)
    V = V/np.sqrt(G)
    Lq =  np.linalg.cholesky(np.eye(3)+np.dot(V, V.T))
    print 'shape'
    print np.dot(V.T,V)
    print np.dot(np.dot(Ksp, np.linalg.inv(Kp)), Ksp.T)
    #print np.linalg.cholesky(np.dot(np.dot(Ksp, np.linalg.inv(Kp)), Ksp.T))
    w1 = y/np.sqrt(G)
    w2 = np.linalg.solve(Lq, np.dot(V,w1))
    print np.dot(w2,w2)
    
    print '---------------'
    Z =np.linalg.solve(Lq, np.linalg.solve(Lp, np.dot(Ksp.T, np.dot(np.linalg.inv(np.diag(G)), y))))
    print np.dot(Z, Z)
    print np.dot(y, np.dot(np.linalg.inv(np.diag(G)), np.dot(Ksp, np.linalg.solve(Lp.T, np.linalg.solve(Lq.T, Z)))))
    Z1 = np.linalg.solve(Lp, np.dot(Ksp.T, np.dot(np.linalg.inv(np.diag(G)), y)))
    print np.dot(np.dot(Z1, np.linalg.inv(np.eye(3)+np.dot(V, V.T))), Z1)
    Z2 = np.dot(Ksp.T, np.dot(np.linalg.inv(np.diag(G)), y))
    print np.dot(np.dot(Z2, np.linalg.inv(np.dot(np.dot(Lp, np.eye(3)+np.dot(V, V.T)), Lp.T))), Z2)
   
    A1 = np.dot(np.dot(Lp, np.eye(3)+np.dot(V, V.T)), Lp.T)
    print A 
    print A1
    
    print 'block mult ----------------'
    ys = y
    yp = np.random.randn(3)
    
    Kp = kernel(Xp)
    Ksp = kernel(Xs, Xp)
    Kps = Ksp.T
    Ks = kernel(Xs)
    
    K = np.vstack((np.hstack((Kp, Kps)), np.hstack((Ksp, Ks))))
    y = np.r_[yp, ys]
    v = np.dot(np.dot(y, K), y)
    v1 = np.dot(np.dot(yp, Kp), yp) + np.dot(np.dot(yp, Kps), ys) + np.dot(np.dot(ys, Ksp), yp) + np.dot(np.dot(ys, Ks), ys)
    v2 =  np.dot(np.dot(yp, Kp), yp) + 2.0*np.dot(np.dot(ys, Ksp), yp) + np.dot(np.dot(ys, Ks), ys)
    print v
    print v1
    print v2
    
    print 'block det----------------'
    Cp = Kp - np.dot(np.dot(Kps, np.linalg.inv(Ks)), Ksp)
    Cs = Ks - np.dot(np.dot(Ksp, np.linalg.inv(Kp)), Kps)
    d = np.linalg.det(K)
    
    d1 = np.linalg.det(Ks) * np.linalg.det(Cp)
    d2 = np.linalg.det(Kp) * np.linalg.det(Cs)
    d3 =  * np.linalg.det(Cp)
    print d
    print d1
    print d2
    
    print 'block inv----------------'
    Cps = -np.dot(np.dot(np.linalg.inv(Kp), Kps), np.linalg.inv(Cs))
    Csp = Cps.T
    v = np.dot(np.dot(y, np.linalg.inv(K)), y)
    v1 = np.dot(np.dot(yp, np.linalg.inv(Cp)), yp) + np.dot(np.dot(yp, Cps), ys) + np.dot(np.dot(ys, Csp), yp) + np.dot(np.dot(ys, np.linalg.inv(Cs)), ys)
    F = np.linalg.inv(Kp) + np.dot(np.dot(np.dot(np.dot(np.linalg.inv(Kp), Kps), np.linalg.inv(Cs)), Ksp), np.linalg.inv(Kp))
    v2 =  np.dot(np.dot(yp, F), yp) + np.dot(np.dot(yp, Cps), ys) + np.dot(np.dot(ys, Csp), yp) + np.dot(np.dot(ys, np.linalg.inv(Cs)), ys)
    print v
    print v1
    print v2