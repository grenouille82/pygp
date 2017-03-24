'''
Created on Sep 5, 2012

@author: marcel
'''
import numpy as np

from upgeo.base.gp import GPRegression
from upgeo.base.infer import ExactInference
from upgeo.eval.trial import CVRegressionExperiment
from upgeo.data.util import loadmat_regdata
from upgeo.base.kernel import ARDSEKernel, NoiseKernel, SEKernel
from upgeo.util.filter import StandardizeFilter, MeanShiftFilter
from numpy.oldnumeric.random_array import permutation
from upgeo.regression.bayes import EMBayesRegression
from sklearn.svm.classes import SVR

def mag_loo_eval(X, y, algo, mag_idx, min_mag=6.0, max_mag=8.0, step_size=0.25, norm_mask=None):
    mag_ranges = np.arange(min_mag, max_mag, step_size)[::-1]
    result = np.empty((len(mag_ranges),2))
    
    hyperparams = algo.hyperparams
    
    k = X.shape[1]+1
    i = 0
    for mag in mag_ranges:
        test = X[:,mag_idx] >= mag
        train = X[:,mag_idx] < mag
        print 'test'
        print np.sum(train)
        print np.sum(test)
        
        Z = np.c_[X,y]
        if norm_mask is None:
            Z = StandardizeFilter().process(Z)
        else:
            Z[:,norm_mask] = StandardizeFilter().process(Z[:,norm_mask])
            print Z[1:5,:]
            
        Xt = Z[:,0:(k-1)]
        yt = Z[:,k-1]
        
        algo.hyperparams = hyperparams
        print Xt[train].shape
        algo.fit(Xt[train], yt[train])
        yhat = algo.predict(Xt[test])
        
        se = (yt[test]-yhat)**2
        print 'mse={0}'.format((np.linalg.norm(yt[test]-yhat)**2)/len(yt[test]))
        result[i,0] = se.mean()
        result[i,1] = se.std()
        i = i+1
        
    algo.hyperparams = hyperparams
    
    print 'Results'
    for i in xrange(len(mag_ranges)):
        print 'Mag={0}: {1}'.format(mag_ranges[i], result[i])


def run_cv(X, y, algo, cov_filter=None, target_filter=None, seed=None):
    if cov_filter:
        X = cov_filter.process(X)
    if target_filter:
        y = cov_filter.process(np.atleast_2d(y))
        y = np.squeeze(y)
        
    
    experiment = CVRegressionExperiment(X, y, 10, seed=seed)
    mse, err = experiment.eval(algo)
    print 'mse={0},err={1}'.format(mse,err)

def run_mag_loo_simdata(X, y, algo):
    X, y = loadmat_regdata('/home/marcel/datasets/multilevel/smsim/eval_smsim_evparams1.mat')
    mag_loo_eval(X, y, algo, mag_idx=4, min_mag=6, max_mag=7.5, step_size=0.25)


 

if  __name__ == '__main__':

    X, y = loadmat_regdata('/home/marcel/datasets/multilevel/smsim/eval_smsim_fullparams_pga2.mat')
    n = len(X)
    #using a subsample of size 1000
    if n > 1000:
        rand = np.random.mtrand.RandomState(837662565132)
        perm = rand.permutation(n)
        X = X[perm[0:1000]]
        y = y[perm[0:1000]]
        
    #print X[1:10]
    #print y[1:10]
    cov_filter = StandardizeFilter(1)
    target_filter = MeanShiftFilter()
    
    kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.1))
    #kernel = MaternKernel(5, np.log(1), np.log(0.5)) + NoiseKernel(np.log(1))
    #kernel = ARDSEKernel(np.log(1)*np.ones(6), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(1)*np.ones(2), np.log(0.5)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
    
    #algo = EMBayesRegression(alpha0=1, beta0=1, weight_bias=True)
    algo = SVR(kernel='rbf')
    #algo = GPRegression(kernel, infer_method=ExactInference)
    run_cv(X, y, algo, cov_filter, target_filter, 42766543542)
    #run_cv_gmmdata(algo, runs=5)
    #run_mag_loo_simdata(algo)
    #run_mag_loo_gmmdata(algo)
    

 