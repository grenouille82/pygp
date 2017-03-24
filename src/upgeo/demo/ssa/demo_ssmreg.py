'''
Created on Feb 4, 2013

@author: marcel
'''
from upgeo.util.filter import MinMaxFilter
import scipy
from upgeo.util.array import unique

'''
Created on Feb 2, 2013

@author: marcel
'''
import numpy as np

import upgeo.util.metric as metric
import upgeo.util.stats as stats

from upgeo.demo.util import loadmat_data, loadmat_folds,\
    loadmat_data_and_periods
from upgeo.base.kernel import SEKernel, NoiseKernel, SqConstantKernel,\
       LinearKernel, ARDSEKernel, RBFKernel, ARDLinearKernel, ARDSELinKernel,\
    ARDRBFKernel, GroupNoiseKernel, MaskedFeatureKernel, HiddenKernel,\
    FixedParameterKernel, CorrelatedNoiseKernel, TaskNoiseKernel,\
    PolynomialKernel, ARDPolynomialKernel
from upgeo.base.selector import KMeansSelector, RandomSubsetSelector
from upgeo.base.gp import GPRegression, SparseGPRegression,\
    PITCSparseGPRegression
from upgeo.base.infer import ExactInference, FITCExactInference,\
    PITCExactInference, PITCOnePassInference, FITCOnePassInference
from upgeo.regression.bayes import EMBayesRegression
from numpy.core.numeric import array_str
from upgeo.regression.linear import LSRegresion
from upgeo.base.mean import BiasedLinearMean, HiddenMean, MaskedFeatureMean

def flatten_data_by_periods(X,Y,periods):
    n = X.shape[0]
    
    Ynan = np.isnan(Y)
    print 'Ynan={0}'.format(Ynan)
    reps = np.sum(~Ynan, 1)
    print 'reps={0}'.format(reps)
    Xret = np.repeat(X, reps, 0)
    yret = np.ravel(Y[~Ynan])
    Xperiods = np.tile(periods, n)
    Xret = np.c_[Xret, Xperiods[np.ravel(~Ynan)]]
    print 'Xret={0}'.format(Xret)
    print 'yret={0}'.format(yret)
    print 'Y={0}'.format(Y)
    return Xret,yret    
    

def eval_reg(train, test, algo, periods):
    Xtest = test[0]
    Ytest = test[1]
    
    Xtrain, ytrain = flatten_data_by_periods(train[0], train[1], periods)
    period_idx = Xtrain.shape[1]-1
    min_period = np.min(periods)
    max_period = np.max(periods)
    Xtrain[:,period_idx] = (Xtrain[:,period_idx]-min_period)/(max_period-min_period) 
    
    algo.fit(Xtrain, ytrain)
    print 'Xtrain'
    print Xtrain
    
    n = Xtest.shape[0]
    d = periods.shape[0]
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
        norm_period = (periods[i]-min_period)/(max_period-min_period)
        m = np.sum(~Ytest_nan[:,i])
        
        print 'Xtest'
        print np.c_[Xtest[~Ytest_nan[:,i]], np.repeat(norm_period, m)]
        yfit, var = algo.predict(np.c_[Xtest[~Ytest_nan[:,i]], np.repeat(norm_period, m)], ret_var=True)
        Yfit[~Ytest_nan[:,i],i] = yfit
        Var[~Ytest_nan[:,i],i] = var 
        
        mse[i] = metric.mspe(Ytest[~Ytest_nan[:,i],i], yfit)
        nmse[i] = mse[i]/np.var(Ytest[~Ytest_nan[:,i],i])
        mll[i] = metric.nlp(Ytest[~Ytest_nan[:,i],i], yfit, var)
        
        print 'fuck'
        print Xtrain[:,period_idx]==norm_period
        print ytrain[Xtrain[:,period_idx]==norm_period]
        nmll[i] = mll[i]-metric.nlpp(Ytest[~Ytest_nan[:,i],i], np.mean(ytrain[Xtrain[:,period_idx]==norm_period]), np.var(ytrain[Xtrain[:,period_idx]==norm_period]))
     
    return mse, nmse, mll, nmll, Yfit, Var

def eval_optsparsegp(train, test, periods, kernel, noise_kernel, selector):
    Xtest = test[0]
    Ytest = test[1]
    
    Xtrain, ytrain = flatten_data_by_periods(train[0], train[1], periods)
    period_idx = Xtrain.shape[1]-1
    min_period = np.min(periods)
    max_period = np.max(periods)
    Xtrain[:,period_idx] = (Xtrain[:,period_idx]-min_period)/(max_period-min_period) 
    
    
    gp = SparseGPRegression(kernel, noise_kernel, infer_method=FITCOnePassInference, selector=selector, fix_inducing=True)
    gp.fit(Xtrain, ytrain)
    #hidden_kernel = HiddenKernel(kernel)
    #hidden_noise_kernel = HiddenKernel(noise_kernel)
    #gp = SparseGPRegression(hidden_kernel, hidden_noise_kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=False)
    #gp.fit(Xtrain, ytrain)
    #gp = SparseGPRegression(kernel, noise_kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=True)
    #gp.fit(Xtrain, ytrain)
    print 'Xtrain'
    print Xtrain
    
    n = Xtest.shape[0]
    d = periods.shape[0]
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
        norm_period = (periods[i]-min_period)/(max_period-min_period)
        m = np.sum(~Ytest_nan[:,i])
        
        #print 'Xtest'
        #print np.c_[Xtest[~Ytest_nan[:,i]], np.repeat(norm_period, m)]
        yfit, var = gp.predict(np.c_[Xtest[~Ytest_nan[:,i]], np.repeat(norm_period, m)], ret_var=True)
        Yfit[~Ytest_nan[:,i],i] = yfit
        Var[~Ytest_nan[:,i],i] = var    
        
        print yfit
        print var
        
        mse[i] = metric.mspe(Ytest[~Ytest_nan[:,i],i], yfit)
        nmse[i] = mse[i]/np.var(Ytest[~Ytest_nan[:,i],i])
        print 'nmse={0}'.format(nmse[i])
        mll[i] = metric.nlp(Ytest[~Ytest_nan[:,i],i], yfit, var)
        
        #print 'fuck'
        #print Xtrain[:,period_idx]==norm_period
        #print ytrain[Xtrain[:,period_idx]==norm_period]
        nmll[i] = mll[i]-metric.nlpp(Ytest[~Ytest_nan[:,i],i], np.mean(ytrain[Xtrain[:,period_idx]==norm_period]), np.var(ytrain[Xtrain[:,period_idx]==norm_period]))
     
    return mse, nmse, mll, nmll, Yfit, Var

    
def eval_optpitcgp(train, test, periods, kernel, noise_kernel, selector):
    Xtest = test[0]
    Ytest = test[1]
    print train[0].shape
    print test[0].shape
    
    Xtrain, ytrain = flatten_data_by_periods(train[0], train[1], periods)
    
    period_idx = Xtrain.shape[1]-1
    min_period = np.min(periods)
    max_period = np.max(periods)

    sortidx = Xtrain[:,period_idx].argsort()
    Xtrain = Xtrain[sortidx]
    ytrain = ytrain[sortidx]
    periodA, groups = unique(Xtrain[:,period_idx], True)
    print 'count'
    print np.sum(Xtrain[:,period_idx] == 1)
    print 'Xtrain'
    print Xtrain[935]
    print Xtrain[:,period_idx]
    print Xtrain
    print Xtrain.shape
    print sortidx
    print groups
    print periodA
    print Xtrain[:,period_idx][groups]
    Xtrain[:,period_idx] = (Xtrain[:,period_idx]-min_period)/(max_period-min_period)
    
    
    gp = PITCSparseGPRegression(groups, kernel, noise_kernel, infer_method=PITCOnePassInference, selector=selector, fix_inducing=True)
    gp.fit(Xtrain, ytrain)
    #hidden_kernel = HiddenKernel(kernel)
    #hidden_noise_kernel = HiddenKernel(noise_kernel)
    #gp = PITCSparseGPRegression(groups, hidden_kernel, hidden_noise_kernel, infer_method=PITCExactInference, selector=selector, fix_inducing=False)
    #gp.fit(Xtrain, ytrain)
    #gp = PITCSparseGPRegression(groups, kernel, noise_kernel, infer_method=PITCExactInference, selector=selector, fix_inducing=True)
    #gp.fit(Xtrain, ytrain)
    print 'Xtrain'
    print Xtrain
    
    n = Xtest.shape[0]
    d = periods.shape[0]
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
        norm_period = (periods[i]-min_period)/(max_period-min_period)
        m = np.sum(~Ytest_nan[:,i])
        
        #print 'Xtest'
        #print np.c_[Xtest[~Ytest_nan[:,i]], np.repeat(norm_period, m)]
        print 'fucker'
        yfit, var = gp.predict(np.c_[Xtest[~Ytest_nan[:,i]], np.repeat(norm_period, m)], ret_var=True, blocks=np.ones(m)*i) #group_id
        print 'lucker'
        print yfit
        print var
        Yfit[~Ytest_nan[:,i],i] = yfit
        Var[~Ytest_nan[:,i],i] = var 
        
        mse[i] = metric.mspe(Ytest[~Ytest_nan[:,i],i], yfit)
        nmse[i] = mse[i]/np.var(Ytest[~Ytest_nan[:,i],i])
        print 'nmse={0}'.format(nmse[i])
        mll[i] = metric.nlp(Ytest[~Ytest_nan[:,i],i], yfit, var)
        
        
        nmll[i] = mll[i]-metric.nlpp(Ytest[~Ytest_nan[:,i],i], np.mean(ytrain[Xtrain[:,period_idx]==norm_period]), np.var(ytrain[Xtrain[:,period_idx]==norm_period]))
     
    return mse, nmse, mll, nmll, Yfit, Var


def create_meanfct(nfeatures, data=None, mask=None):
    meanfct = None
    
    if data != None:
        rmodel = LSRegresion()
        rmodel.fit(data[0], data[1])
        meanfct = BiasedLinearMean(rmodel.weights, rmodel.intercept)
        meanfct = HiddenMean(meanfct)
    else:
        meanfct = BiasedLinearMean(np.zeros(nfeatures), 0)
    
    if mask != None:
        meanfct = MaskedFeatureMean(meanfct, mask)

    return meanfct

def create_noise_kernel(grp_idx, s, kernel=None, mask=None):
    noise_kernel = GroupNoiseKernel(grp_idx, s)
    if kernel != None:
        noise_kernel = HiddenKernel(noise_kernel)
        noise_kernel = noise_kernel*kernel
    if mask != None:
        noise_kernel = MaskedFeatureKernel(noise_kernel, mask)
    return noise_kernel

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
    nperiods = len(periods)
     
    
    Yfit = np.zeros(Y.shape)
    Var = np.zeros(Y.shape)
        
    mse = np.empty((nfolds,nperiods))
    mll = np.empty((nfolds,nperiods))
    nmse = np.empty((nfolds,nperiods))
    nmll = np.empty((nfolds,nperiods))
    weights = np.empty(nfolds)
    
    
    
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
        
        mask_periods = np.r_[np.zeros(7),1]
        mask_periods = np.array(mask_periods, dtype=np.bool)
        
        #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(0.001)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), mask_periods) + NoiseKernel(np.log(0.5))
        #kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), ~mask_periods) + MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), mask_periods) + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + MaskedFeatureKernel(SqConstantKernel(np.log(1)) * LinearKernel(), ~mask_periods) + NoiseKernel(np.log(0.5)) 
        #kernel = SEKernel(np.log(1), np.log(1)) + MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), mask_periods) + MaskedFeatureKernel(SqConstantKernel(np.log(1)) * LinearKernel(), ~mask_periods) + NoiseKernel(np.log(0.5))
        #kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), ~mask_periods) + MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), mask_periods) + MaskedFeatureKernel(SqConstantKernel(np.log(1)) * LinearKernel(), ~mask_periods) + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5)) + PolynomialKernel(2, np.log(1), np.log(1))
        #kernel = RBFKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = RBFKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.r_[np.log(l), np.log(1)], np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = MaskedFeatureKernel(ARDSEKernel(np.log(1)*np.ones(7), np.log(1)), ~mask_periods) + MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), mask_periods) + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.r_[np.log(l), np.log(1)], np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(8), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.r_[np.log(1)*np.ones(7), np.log(1)], np.log(1)) + MaskedFeatureKernel(ARDLinearKernel(np.log(1)*np.ones(7), np.log(1)), ~mask_periods) + NoiseKernel(np.log(0.5))
        #kernel = MaskedFeatureKernel(ARDSEKernel(np.log(1)*np.ones(7), np.log(1)), ~mask_periods) + MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), mask_periods) + MaskedFeatureKernel(ARDLinearKernel(np.log(1)*np.ones(7), np.log(1)), ~mask_periods) + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(1)*np.ones(8), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(l), np.log(1)) + ARDPolynomialKernel(3, np.log(1)*np.ones(len(l)), np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDSELinKernel(np.log(l), np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDRBFKernel(np.log(l), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDRBFKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        selector = RandomSubsetSelector(15) 
        #selector = KMeansSelector(15, False)
        
        #kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask) + CorrelatedNoiseKernel(0, np.log(0.1), np.log(0.5))
        #kernel = MaskedFeatureKernel(ARDSEKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask) + CorrelatedNoiseKernel(0, np.log(0.1), np.log(0.5))
        
        #meanfunctions for standard data
        #meanfct = create_meanfct(7, data=None, mask=None) #mean
        #meanfct = create_meanfct(7, data=data_train, mask=None) #fixmean
        
        #meanfunctions for different parameters in the meanfct and covfct
        #meanfct = create_meanfct(10, data=None, mask=None) #mean
        #meanfct = create_meanfct(10, data=data_train, mask=None) #fixmean
        #kernel = MaskedFeatureKernel(kernel, fmask)
        
        #create complex noise model
        #noise_kernel = create_noise_kernel(0, np.log(1))
        #noise_kernel = create_noise_kernel(0, np.log(1), MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[np.zeros(6), np.ones(2)], dtype=np.bool)))
        #noise_kernel = create_noise_kernel(0, np.log(1), MaskedFeatureKernel(ARDSEKernel(np.log(l[6:7]), np.log(1)), np.array(np.r_[np.zeros(6), np.ones(2)], dtype=np.bool)))
        #kernel = MaskedFeatureKernel(kernel, fmask) + noise_kernel
                
        #mtl kernel
        #noise_kernel = NoiseKernel(np.log(0.5)) #+ TaskNoiseKernel(X[train,0], 0, np.log(0.001))
        #mtl_kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[0, np.ones(5), np.zeros(2)] ,dtype=bool))*MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[0, np.zeros(5), np.ones(2)] ,dtype=bool))
        #mtl_kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[0, np.ones(5), np.zeros(2)] ,dtype=bool))*MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[0, np.ones(7)] ,dtype=bool))
        #mtl_kernel = mtl_kernel + MaskedFeatureKernel(SqConstantKernel(np.log(1)) * LinearKernel(), fmask)
        #kernel = FixedParameterKernel(mtl_kernel+noise_kernel, [3])
        
        kernel = SEKernel(np.log(1), np.log(1)) + MaskedFeatureKernel(SqConstantKernel(np.log(1)) * LinearKernel(), ~mask_periods)
        #kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel(), ~mask_periods) + MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), mask_periods) #+ NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.r_[np.log(l), np.log(1)], np.log(1)) + MaskedFeatureKernel(ARDLinearKernel(np.log(1)*np.ones(7), np.log(1)), ~mask_periods)
        #kernel = ARDSEKernel(np.r_[np.log(l), np.log(1)], np.log(1)) + MaskedFeatureKernel(SqConstantKernel(np.log(1)) * LinearKernel(), ~mask_periods)
        #kernel = MaskedFeatureKernel(ARDSEKernel(np.log(l), np.log(1))+ARDLinearKernel(np.log(1)*np.ones(7), np.log(1)), ~mask_periods) + MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), mask_periods) 
        #noise_kernel = TaskNoiseKernel((periods-np.min(periods))/(np.max(periods)-np.min(periods)), 7, np.log(0.5))
        #noise_kernel = TaskNoiseKernel((periods-np.min(periods))/(np.max(periods)-np.min(periods)), 7, np.log(0.5))
        noise_kernel = NoiseKernel(np.log(0.5))
        #kernel = kernel+noise_kernel
        
        
        
        #algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=True)
        #algo = GPRegression(kernel, infer_method=ExactInference)
        #algo = GPRegression(kernel, meanfct=meanfct, infer_method=ExactInference)
        weights[i] = len(test)
        #mse[i], nmse[i], mll[i], nmll[i], Yfit[test], Var[test] = eval_reg(data_train, data_test, algo, periods)
        #mse[i], nmse[i], mll[i], nmll[i], Yfit[test], Var[test] = eval_optsparsegp(data_train, data_test, periods, kernel, noise_kernel, selector)
        mse[i], nmse[i], mll[i], nmll[i], Yfit[test], Var[test] = eval_optpitcgp(data_train, data_test, periods, kernel, noise_kernel, selector)
        
        
        #print 'hyperparams={0}'.format(kernel.        params)
        print 'task({0}): mse={1},{2}, mll={3},{4}'.format(i, mse[i], nmse[i], mll[i], nmll[i])
        
    for i in xrange(nperiods):    
        print 'CV Results for period={0}:'.format(periods[i])
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
