'''
Created on Oct 11, 2012

@author: marcel
'''
import numpy as np

from numpy.linalg.linalg import LinAlgError
from scipy.linalg.decomp_cholesky import cho_solve

def jit_cholesky(A, max_iter=10, ret_jitter=False):
    '''
    '''
    jitter = 0
    
    try:
        L = np.linalg.cholesky(A)
    except LinAlgError:
        jitter = np.abs(np.mean(np.diag(A)))*1e-6
        n = A.shape[0]
        i = 1
        while i < max_iter:
            try:
                L = np.linalg.cholesky(A + np.diag(np.ones(n)*jitter))
                break
            except LinAlgError:
                jitter *= 10        
            i = i+1 
        if i == max_iter:
            raise LinAlgError()
    
    return L if ret_jitter == False else L, jitter
   
   
def pdinv(A, L=None, ret_jitter):
    if L == None:
        L, jitter = jit_cholesky(A, ret_jitter=True)
    
    Ainv = cho_solve(L, np.eye(A.shape[0]))
    return Ainv, L if ret_jitter == False else Ainv, L, jitter