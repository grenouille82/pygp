'''
Created on May 30, 2013

@author: marcel
'''

import numpy as np
import upgeo.util.metric as metric
import upgeo.util.stats as stats

from upgeo.util.array import unique
from upgeo.demo.util import loadmat_data, loadmat_folds, loadmat_mtl_data
from numpy.core.numeric import array_str
from upgeo.regression.bayes import EMBayesRegression
from upgeo.base.kernel import SEKernel, SqConstantKernel, LinearKernel,\
    NoiseKernel, RBFKernel, ExpGaussianKernel, CompoundKernel,\
    DiracConvolvedKernel, FixedParameterKernel
from upgeo.base.gp import GPRegression
from upgeo.base.infer import ExactInference
from upgeo.util.filter import MeanShiftFilter, MinMaxFilter, FunctionFilter,\
    CompositeFilter
from upgeo.mtl.kernel import ConvolvedMTLKernel
from upgeo.mtl.infer import SparseCMOGPExactInference, CMOGPExactInference
from upgeo.mtl.gp import SparseCMOGPRegression, CMOGPRegression
from upgeo.util.glob import APPROX_TYPE
from upgeo.base.selector import KMeansSelector, FixedSelector

def eval_regression(train, test, task_ids, gp):
    Xtrain = train[0]
    Ytrain = train[1]
    Gtrain = train[2]
    _,itask = unique(Gtrain,True)
    
    Xtest = test[0]
    Ytest = test[1]
    Gtest = test[2]
    
    k = len(task_ids)
    
    n = Ytest.shape[0]
    Yfit = np.zeros(n)
    Var = np.zeros(n)
    
    
    gp.fit(Xtrain, Ytrain, itask)
    print 'opthyperparams={0}'.format(np.exp(gp.hyperparams))
    
    for i in xrange(k):
        #norm_period = (periods[i]-min_periop)/(max_period-min_period)
        #m = np.sum(~Ytest_nan[:,i])
        
        train_ids = Gtrain == task_ids[i]
        test_ids = Gtest == task_ids[i]
        
        yfit, var = gp.predict_task(Xtest[test_ids], q=i, ret_var=True)
     
        print 'yfit={0}'.format(yfit)
        print 'var={0}'.format(var)
     
        Yfit[test_ids] = yfit
        Var[test_ids] = var
        
    mse = metric.mspe(Ytest, Yfit)
    nmse = mse/np.var(Ytest)
        
    mll = metric.nlp(Ytest, Yfit, Var)
    nmll = mll-metric.nlpp(Ytest, np.mean(Ytrain), np.var(Ytrain))
     
    return mse, nmse, mll, nmll, Yfit, Var

if __name__ == '__main__':
    nfolds = 10
    fold_filename = '//home/marcel/datasets/multilevel/kappa/splitz/kappa_{0}_indexes.mat'
    
    filename = '/home/marcel/datasets/multilevel/kappa/eval_kappa_mtl.mat'
    
    X,y,tasks = loadmat_mtl_data(filename)
    X = X[:,np.newaxis]
    #X = np.c_[X[:,0], X[:,0]]
    task_ids = unique(tasks)
    k = len(task_ids)
    #for simple pitc approximation
    #X = X[:,1:]
    #print Y
    #choose period of the response spectra as target variable
    #print periods == 0.0
    #print period_idx
    
    print 'X={0}'.format(X)
    print X.shape
    
    n = X.shape[0]
    ypred = np.zeros((n,2)) #matrix of the fitted value and its variance
        
    mse = np.empty(nfolds)
    mll = np.empty(nfolds)
    nmse = np.empty(nfolds)
    nmll = np.empty(nfolds)
    weights = np.empty(nfolds)
        
    
    
    for i in xrange(nfolds):
        train, test = loadmat_folds(fold_filename.format(i+1))
    
        cov_filter = CompositeFilter([FunctionFilter(np.log, np.exp), MinMaxFilter()])
        #cov_filter = MinMaxFilter()
        target_filter = CompositeFilter([FunctionFilter(np.log, np.exp), MeanShiftFilter()])
        #target_filter = MeanShiftFilter()

        Xtrain = cov_filter.process(X[train])
        ytrain = np.squeeze(target_filter.process(y[train][:,np.newaxis]))
        Xtest = cov_filter.process(X[test],True)
        ytest = np.squeeze(target_filter.process(y[test][:,np.newaxis],True))
    
        data_train = (Xtrain, ytrain, tasks[train])
        data_test = (Xtest, ytest, tasks[test])
        print data_train
        
        #mask for selecting base features for the cov function 
        #fmask = np.r_[1, 0, np.ones(5), 0, 0, 1] #distance  (data2 features)
        #fmask = np.r_[1, 0, np.ones(4), 0, 1, 0, 1] #log distance (data3 features)
        #fmask = np.r_[0, np.ones(7)]
        #fmask = np.array(fmask, dtype=np.bool)
        
        latent_kernel = ExpGaussianKernel(np.log(0.1))
        #latent_kernel = CompoundKernel([ExpGaussianKernel(np.log(0.1)), ExpGaussianKernel(np.log(0.2))])
        #latent_kernel = DiracConvolvedKernel(FixedParameterKernel(SEKernel(np.log(0.1),np.log(1)), [1]))
        #noise_kernel = SEKernel(np.log(0.1), np.log(1)) + NoiseKernel(np.log(0.5))
        noise_kernel = SEKernel(np.log(0.1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        
        theta = [np.log(0.1), np.log(0.1)]
        #theta = [np.log(0.1), np.log(0.1), np.log(0.2), np.log(0.1)]
        #theta = [np.log(0.1)]
        kernel = ConvolvedMTLKernel(latent_kernel, theta, k, noise_kernel)
        
        
        selector = KMeansSelector(15, False) 
        Xu = selector.apply(data_train[0], data_train[1])
        selector = FixedSelector(Xu)
        
        #algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=False)
        #gp = SparseCMOGPRegression(kernel, infer_method=SparseCMOGPExactInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=True)
        gp = CMOGPRegression(kernel, infer_method=CMOGPExactInference)

        #algo = PITCSparseGPRegression(igroup, kernel, noise_kernel, infer_method=PITCExactInference, selector=selector, fix_inducing=False)
        #algo = GPRegression(kernel, meanfct=meanfct, infer_method=ExactInference)
        weights[i] = len(test)
        
        mse[i], nmse[i], mll[i], nmll[i], yfit, var = eval_regression(data_train, data_test, task_ids, gp)
        #mse[i], nmse[i], mll[i], nmll[i], yfit, var = eval_optsparsegp(data_train, data_test, kernel, noise_kernel, selector, igroup)
        ypred[test,0] = yfit
        ypred[test,1] = var
        
        
        #print 'hyperparams={0}'.format(kernel.params)
        print 'task({0}): mse={1},{2}, mll={3},{4}'.format(i, mse[i], nmse[i], mll[i], nmll[i])
        
    print 'CV Results:'
    print 'mse'
    print array_str(mse, precision=16)
    print 'nmse'
    print array_str(nmse, precision=16)
    print 'mll'
    print array_str(mll, precision=16)
    print 'nmll'
    print array_str(nmll, precision=16)
    print 'Total Results:'
    
    print 'Output Result:{0}'.format(i) 
    means = np.asarray([stats.mean(mse, weights), stats.mean(nmse, weights), 
                        stats.mean(mll, weights), stats.mean(nmll, weights)])
    std = np.asarray([stats.stddev(mse, weights), stats.stddev(nmse, weights),
                      stats.stddev(mll, weights), stats.stddev(nmll, weights)])
        
    print 'mean={0}'.format(array_str(means, precision=16))
    print 'err={0}'.format(array_str(std, precision=16))
        
    #np.savetxt('/home/mhermkes/datasets/multilevel/nga/ssa/results/pga/gpsepoly2_grpnoise_data2', ypred, delimiter=',')
