'''
Created on Mar 12, 2013

@author: marcel
'''
import numpy as np

from upgeo.demo.util import loadmat_data_and_periods, loadmat_folds,\
    prepare_mtl_data
from upgeo.base.selector import RandomSubsetSelector, FixedSelector
from upgeo.base.kernel import NoiseKernel, SEKernel, LinearKernel,\
    SqConstantKernel, ExpGaussianKernel, DiracConvolvedKernel
from upgeo.mtl.kernel import ConvolvedMTLKernel
from upgeo.mtl.gp import SparseCMOGPRegression
from upgeo.util.glob import APPROX_TYPE
from upgeo.mtl.infer import SparseCMOGPExactInference,\
    SparseCMOGPOnePassInference
from numpy.core.numeric import array_str
from upgeo.util import stats, metric

def eval_mogp(train, test, algo, period_idxs):
    Xtest = test[0]
    Ytest = test[1]
    
    Xtrain, ytrain, itask = prepare_mtl_data(train[0], train[1])
    #period_idx = Xtrain.shape[1]-1
    #min_period = np.min(periods)
    #max_period = np.max(periods)
    #Xtrain[:,period_idx] = (Xtrain[:,period_idx]-min_period)/(max_period-min_period) 
    
    #print 'Xtrain={0}'.format(Xtrain)
    #print 'ytrain={0}'.format(ytrain)
    #print 'itask={0}'.format(itask)
    algo.fit(Xtrain, ytrain, itask)
    
    n = Xtest.shape[0]
    d = period_idxs.shape[0]
    Ytest = Ytest[:,period_idxs]
    Ytest_nan = np.isnan(Ytest)
    Yfit = np.zeros((n,d))
    Var = np.zeros((n,d))
    
    Yfit[Ytest_nan] = np.nan
    Var[Ytest_nan] = np.nan
    
    mse = np.empty(d)
    nmse = np.empty(d)
    mll = np.empty(d)
    nmll = np.empty(d)
    
    for i in xrange(d):
        #norm_period = (periods[i]-min_periop)/(max_period-min_period)
        #m = np.sum(~Ytest_nan[:,i])
        q = period_idxs[i]
        
        start = itask[q]
        end = itask[q+1] if q < d-1 else len(Xtrain)
        
        yfit, var = algo.predict_task(Xtest[~Ytest_nan[:,i]], q, ret_var=True)
        #print 'Xtest'
        #print np.c_[Xtest[~Ytest_nan[:,i]], np.repeat(norm_period, m)]
        #yfit, var = algo.predict(np.c_[Xtest[~Ytest_nan[:,i]], np.repeat(norm_period, m)], ret_var=True)
        Yfit[~Ytest_nan[:,i],i] = yfit
        Var[~Ytest_nan[:,i],i] = var    
        print 'yfit={0}'.format(yfit)
        print 'var={0}'.format(var)
        
        mse[i] = metric.mspe(Ytest[~Ytest_nan[:,i],i], yfit)
        nmse[i] = mse[i]/np.var(Ytest[~Ytest_nan[:,i],i])
        
        mll[i] = metric.nlp(Ytest[~Ytest_nan[:,i],i], yfit, var)
        nmll[i] = mll[i]-metric.nlpp(Ytest[~Ytest_nan[:,i],i], np.mean(ytrain[start:end]), np.var(ytrain[start:end]))
     
    return mse, nmse, mll, nmll, Yfit, Var


if __name__ == '__main__':
    nfolds = 5
    fold_filename = '//home/marcel/datasets/multilevel/nga/ssa/splitz/nga_{0}_indexes.mat'
    
    filename = '/home/marcel/datasets/multilevel/nga/ssa/ssa_eval.mat'
    
    X,Y,periods = loadmat_data_and_periods(filename)
    #Y = Y[:,0:5]
    #periods = periods[0:5]
    #print Y
    #choose period of the response spectra as target variable
    #print periods == 0.0
    #period_idx = np.flatnonzero(periods == 8)
    #print period_idx
    #y = np.squeeze(Y[:,period_idx])
    
    n = X.shape[0]
    period_idx = np.asarray([np.nonzero(periods == p)[0][0] for p in [0, 0.1, 0.5, 4, 8]])
    nperiods = len(periods)
    m = len(period_idx)
     
    
    Yfit = np.zeros((Y.shape[0],m))
    Var = np.zeros((Y.shape[0],m))
        
    mse = np.empty((nfolds,m))
    mll = np.empty((nfolds,m))
    nmse = np.empty((nfolds,m))
    nmll = np.empty((nfolds,m))
    weights = np.empty(nfolds)
    
    
    print 'period_idx={0}'.format(period_idx)
    
    
    for i in xrange(nfolds):
        train, test = loadmat_folds(fold_filename.format(i+1))
        
        data_train = (X[train], Y[train])
        data_test = (X[test], Y[test])
        
        #mask for selecting base features for the cov function 
        #fmask = np.r_[1, 0, np.ones(5), 0, 0, 1] #distance  (data2 features)
        #fmask = np.r_[1, 0, np.ones(4), 0, 1, 0, 1] #log distance (data3 features)
        fmask = np.r_[0, np.ones(8)]
        fmask = np.array(fmask, dtype=np.bool)
        
        #init lenght-scales
        l = (np.max(data_train[0],0)-np.min(data_train[0],0))/2
        #l = (np.max(data_train[0][:,fmask],0)-np.min(data_train[0][:,fmask],0))/2
        l[l == 0] = 1e-4
        print 'l={0}'.format(l)
        
        
        
        selector = RandomSubsetSelector(15) 
        Xu = selector.apply(data_train[0], data_train[1])
        selector = FixedSelector(Xu)
        #selector = KMeansSelector(15, False)
        
        
        latent_kernel = ExpGaussianKernel(np.log(0.01))
        #latent_kernel = DiracConvolvedKernel(SEKernel(np.log(1),np.log(0.1)))
        #latent_Kernel = DiracConvolvedKernel(GaussianKernel(np.log(1)))
        #noise_kernel = SEKernel(np.log(1), np.log(0.1)) + SqConstantKernel(np.log(0.1)) * LinearKernel() + NoiseKernel(np.log(1))
        noise_kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(02))
        #noise_kernel = TaskNoiseKernel((periods-np.min(periods))/(np.max(periods)-np.min(periods)), 7, np.log(0.5))
        #noise_kernel = TaskNoiseKernel((periods-np.min(periods))/(np.max(periods)-np.min(periods)), 7, np.log(0.5))
        
        theta = [np.log(0.01), np.log(1)]
        #theta = [np.log(1)]
        kernel = ConvolvedMTLKernel(latent_kernel, theta, nperiods, noise_kernel)
        
        
        #algo = SparseCMOGPRegression(kernel, infer_method=SparseCMOGPExactInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=False)
        algo = SparseCMOGPRegression(kernel, infer_method=SparseCMOGPOnePassInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=False)
        
        #algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=True)
        #algo = GPRegression(kernel, infer_method=ExactInference)
        #algo = GPRegression(kernel, meanfct=meanfct, infer_method=ExactInference)
        weights[i] = len(test)
        #mse[i], nmse[i], mll[i], nmll[i], Yfit[test], Var[test] = eval_reg(data_train, data_test, algo, periods)
        #mse[i], nmse[i], mll[i], nmll[i], Yfit[test], Var[test] = eval_optsparsegp(data_train, data_test, periods, kernel, noise_kernel, selector)
        mse[i], nmse[i], mll[i], nmll[i], Yfit[test], Var[test] = eval_mogp(data_train, data_test, algo, period_idx)
        
        
        #print 'hyperparams={0}'.format(kernel.        params)
        print 'task({0}): mse={1},{2}, mll={3},{4}'.format(i, mse[i], nmse[i], mll[i], nmll[i])
        
    for i in xrange(m):    
        print 'CV Results for period={0}:'.format(periods[period_idx[i]])
        print 'mse'     
        print array_str(mse[:,i], precision=16)
        print 'nmse'
        print array_str(nmse[:,i], precision=16)
        print 'mll'
        print array_str(mll[:,i], precision=16)
        print 'nmll'
        print array_str(nmll[:,i], precision=16)
        print 'Total Results:'
        
        print 'Output Result:{0}'.format(i) 
        means = np.asarray([stats.mean(mse[:,i], weights), stats.mean(nmse[:,i], weights), 
                            stats.mean(mll[:,i], weights), stats.mean(nmll[:,i], weights)])
        std = np.asarray([stats.stddev(mse[:,i], weights), stats.stddev(nmse[:,i], weights),
                          stats.stddev(mll[:,i], weights), stats.stddev(nmll[:,i], weights)])
            
        print 'mean={0}'.format(array_str(means, precision=16))
        print 'err={0}'.format(array_str(std, precision=16))
        
    #np.savetxt('/home/mhermkes/datasets/multilevel/nga/ssa/results/pga/gpsepoly2_grpnoise_data2', ypred, delimiter=',')
