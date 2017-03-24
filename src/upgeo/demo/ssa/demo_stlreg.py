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
    PITCExactInference
from upgeo.regression.bayes import EMBayesRegression
from numpy.core.numeric import array_str
from upgeo.regression.linear import LSRegresion
from upgeo.base.mean import BiasedLinearMean, HiddenMean, MaskedFeatureMean
from upgeo.util.array import unique

def eval_reg(train, test, algo):
    Xtrain = train[0]
    Ytrain = train[1]

    Xtest = test[0]
    Ytest = test[1]
 
    algo.fit(Xtrain, Ytrain)
    yfit, var = algo.predict(Xtest, ret_var=True)
    print 'hyperparams={0}'.format(algo.hyperparams)
    
    mse = metric.mspe(Ytest, yfit)
    nmse = mse/np.var(Ytest)
    mll = metric.nlp(Ytest, yfit, var)
    nmll = mll-metric.nlpp(Ytest, np.mean(Ytrain), np.var(Ytrain))
     
    return mse, nmse, mll, nmll, yfit, var


def eval_optsparsegp(train, test, kernel, noise_kernel, selector, groups=None):
    Xtrain = train[0]
    Ytrain = train[1]

    Xtest = test[0]
    Ytest = test[1]

    if groups == None: 
        gp = SparseGPRegression(kernel, noise_kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=True)
        gp.fit(Xtrain, Ytrain)
        hidden_kernel = HiddenKernel(kernel)
        hidden_noise_kernel = HiddenKernel(noise_kernel)
        gp = SparseGPRegression(hidden_kernel, hidden_noise_kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=False)
        gp.fit(Xtrain, Ytrain)
        gp = SparseGPRegression(kernel, noise_kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=True)
        gp.fit(Xtrain, Ytrain)
    else:
        gp = PITCSparseGPRegression(groups, kernel, noise_kernel, infer_method=PITCExactInference, selector=selector, fix_inducing=True)
        gp.fit(Xtrain, Ytrain)
        hidden_kernel = HiddenKernel(kernel)
        hidden_noise_kernel = HiddenKernel(noise_kernel)
        gp = PITCSparseGPRegression(groups, hidden_kernel, hidden_noise_kernel, infer_method=PITCExactInference, selector=selector, fix_inducing=False)
        gp.fit(Xtrain, Ytrain)
        gp = PITCSparseGPRegression(groups, kernel, noise_kernel, infer_method=PITCExactInference, selector=selector, fix_inducing=True)
        gp.fit(Xtrain, Ytrain)

    
    yfit, var = gp.predict(Xtest, ret_var=True)
    
    mse = metric.mspe(Ytest, yfit)
    nmse = mse/np.var(Ytest)
    mll = metric.nlp(Ytest, yfit, var)
    nmll = mll-metric.nlpp(Ytest, np.mean(Ytrain), np.var(Ytrain))
     
    return mse, nmse, mll, nmll, yfit, var


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
    
    filename = '/home/marcel/datasets/multilevel/nga/ssa/ssa_eval_eq.mat'
    
    X,Y,periods = loadmat_data_and_periods(filename)
    #for simple pitc approximation
    eqs = X[:,0]
    #X = X[:,1:]
    #print Y
    #choose period of the response spectra as target variable
    #print periods == 0.0
    period_idx = np.flatnonzero(periods == 8)
    #print period_idx
    y = np.squeeze(Y[:,period_idx])
   
    mvs = np.isnan(y) #missing values of the target
    
    n = X.shape[0]
    ypred = np.zeros((n,2)) #matrix of the fitted value and its variance
    ypred[mvs,:] = np.nan
        
    mse = np.empty(nfolds)
    mll = np.empty(nfolds)
    nmse = np.empty(nfolds)
    nmll = np.empty(nfolds)
    weights = np.empty(nfolds)
    
    
    
    
    
    for i in xrange(nfolds):
        train, test = loadmat_folds(fold_filename.format(i+1))
        mvs_train = np.isnan(y[train])
        mvs_test = np.isnan(y[test])
    
        data_train = (X[train[~mvs_train]], y[train[~mvs_train]])
        data_test = (X[test[~mvs_test]], y[test[~mvs_test]])
        
        _, igroup = unique(eqs[~mvs_train], True)
        
        print data_train
        
        #mask for selecting base features for the cov function 
        #fmask = np.r_[1, 0, np.ones(5), 0, 0, 1] #distance  (data2 features)
        #fmask = np.r_[1, 0, np.ones(4), 0, 1, 0, 1] #log distance (data3 features)
        fmask = np.r_[0, np.ones(7)]
        fmask = np.array(fmask, dtype=np.bool)
        
        #init lenght-scales
        #l = (np.max(data_train[0],0)-np.min(data_train[0],0))/2
        l = (np.max(data_train[0][:,fmask],0)-np.min(data_train[0][:,fmask],0))/2
        l[l == 0] = 1e-4
        #print 'l={0}'.format(l)
        
        
        #algo = EMBayesRegression(alpha0=1, beta0=1, weight_bias=True)
        kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel()# + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(0.001)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + PolynomialKernel(2, np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5)) + PolynomialKernel(2, np.log(1), np.log(1))
        #kernel = RBFKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = RBFKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(l), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(1)*np.ones(7), np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(len(l)), np.log(1)) #+ NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(l), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(l), np.log(1)) + ARDPolynomialKernel(2, np.log(1)*np.ones(len(l)), np.log(1), np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(len(l)), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDSELinKernel(np.log(l), np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDRBFKernel(np.log(l), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDRBFKernel(np.log(l), np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(len(l)), np.log(1))  + NoiseKernel(np.log(0.5))
        selector = RandomSubsetSelector(30) 
        
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
        noise_kernel = create_noise_kernel(0, np.log(1)) + NoiseKernel(np.log(0.5))
        #noise_kernel = create_noise_kernel(0, np.log(1), MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[np.zeros(6), np.ones(2)], dtype=np.bool)))
        #noise_kernel = create_noise_kernel(0, np.log(1), MaskedFeatureKernel(ARDSEKernel(np.log(l[6:7]), np.log(1)), np.array(np.r_[np.zeros(6), np.ones(2)], dtype=np.bool)))
        kernel = MaskedFeatureKernel(kernel, fmask) + noise_kernel
        
        #mtl kernel
        #noise_kernel = NoiseKernel(np.log(0.5)) #+ TaskNoiseKernel(X[train,0], 0, np.log(0.001))
        #mtl_kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[0, np.ones(5), np.zeros(2)] ,dtype=bool))*MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[0, np.zeros(5), np.ones(2)] ,dtype=bool))
        #mtl_kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[0, np.ones(5), np.zeros(2)] ,dtype=bool))*MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[0, np.ones(7)] ,dtype=bool))
        #mtl_kernel = mtl_kernel + MaskedFeatureKernel(SqConstantKernel(np.log(1)) * LinearKernel(), fmask)
        #kernel = FixedParameterKernel(mtl_kernel+noise_kernel, [3])
        
        #algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=False)
        algo = GPRegression(kernel, infer_method=ExactInference)
        #algo = PITCSparseGPRegression(igroup, kernel, noise_kernel, infer_method=PITCExactInference, selector=selector, fix_inducing=False)
        #algo = GPRegression(kernel, meanfct=meanfct, infer_method=ExactInference)
        weights[i] = len(test)
        
        mse[i], nmse[i], mll[i], nmll[i], yfit, var = eval_reg(data_train, data_test, algo)
        #mse[i], nmse[i], mll[i], nmll[i], yfit, var = eval_optsparsegp(data_train, data_test, kernel, noise_kernel, selector, igroup)
        ypred[test[~mvs_test],0] = yfit
        ypred[test[~mvs_test],1] = var
        
        
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
