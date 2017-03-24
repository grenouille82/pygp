'''
Created on May 28, 2013

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
    NoiseKernel, RBFKernel
from upgeo.base.gp import GPRegression
from upgeo.base.infer import ExactInference
from upgeo.util.filter import MeanShiftFilter, MinMaxFilter, FunctionFilter,\
    CompositeFilter

def eval_stlalgo(train, test, itasks, model):
    Xtrain = train[0]
    Ytrain = train[1]

    Xtest = test[0]
    Ytest = test[1]
    
    ntasks = len(itasks)
    n = Ytest.shape[0]
     
    params = np.copy(model.hyperparams)
     
    smse = np.zeros(ntasks)
    msll = np.zeros(ntasks)
    itasks = np.r_[itasks, len(Ytrain)]
    for i in xrange(ntasks):
        start = itasks[i]
        end = itasks[i+1]
        
        model.hyperparams = params
        model.fit(Xtrain[start:end], Ytrain[start:end])
        yfit, var = model.predict(Xtest, ret_var=True)
        
        smse[i] = metric.mspe(Ytest[:,i], yfit)/np.var(Ytest[:,i])
        print 'fuck'
        print metric.mspe(Ytest[:,i], yfit)
        mll = np.sum((yfit-Ytest[:,i])**2/(2*var) + 0.5*np.log(2*np.pi*var))/n
        nmll = (np.sum((np.mean(Ytrain[start:end])-Ytest[:,i])**2)/n)/(2.0*np.var(Ytrain[start:end]))+0.5*np.log(2*np.pi*np.var(Ytrain[start:end]))
        #print yfit[:,i]
        print 'suck'
        print mll
        print nmll
        #print metric.nlp(Ytest[:,i], yfit[:,i], var[:,i])
        #print np.sum((yfit[:,i]-Ytest[:,i])**2/(2*var[:,i]) + 0.5*np.log(2*np.pi*var[:,i]))/n
        #print (np.sum((np.mean(Ytrain[start:end])-Ytest[:,i])**2)/n)/(2.0*np.var(Ytrain[start:end]))+0.5*np.log(2*np.pi*np.var(Ytrain[start:end]))
        #print ((np.mean(Ytrain[start:end])-np.mean(Ytest[:,i]))**2+np.var(Ytest[:,i]))/(2*np.var(Ytrain[start:end])) + 0.5*np.log(2*np.pi*np.var(Ytrain[start:end])) 
        #print 1/2*np.log(2*np.pi*np.var(Ytrain[start:end]))
        #print 0.5*np.log(2*np.pi*np.var(Ytrain[start:end]))
        msll[i] = mll-nmll
    
    return (smse, msll)


def eval_regression(train, test, task_ids, model):
    Xtrain = train[0]
    Ytrain = train[1]
    Ttrain = train[2]
    
    Xtest = test[0]
    Ytest = test[1]
    Ttest = test[2]
    
    k = len(task_ids)
    
    n = Ytest.shape[0]
    Yfit = np.zeros(n)
    Var = np.zeros(n)
    
    params = np.copy(model.hyperparams)
    for i in xrange(k):
        #norm_period = (periods[i]-min_periop)/(max_period-min_period)
        #m = np.sum(~Ytest_nan[:,i])
        
        train_ids = Ttrain == task_ids[i]
        test_ids = Ttest == task_ids[i]
        
        #algo = EMBayesRegression(alpha0=1, beta0=1, weight_bias=True)
        model.hyperparams = params
        model.fit(Xtrain[train_ids], Ytrain[train_ids])
        yfit, var = model.predict(Xtest[test_ids], ret_var=True)
     
        Yfit[test_ids] = yfit
        Var[test_ids] = var
        
        print 'sum_testdata={0}'.format(np.sum(test_ids))
        
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
    
        #cov_filter = CompositeFilter([FunctionFilter(np.log, np.exp), MinMaxFilter()])
        cov_filter = MinMaxFilter()
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
        
        #init lenght-scales
        
        
        model = EMBayesRegression(alpha0=1, beta0=1, weight_bias=True)
        #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
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
        #selector = RandomSubsetSelector(30) 
        
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
        #noise_kernel = create_noise_kernel(0, np.log(1)) + NoiseKernel(np.log(0.5))
        #noise_kernel = create_noise_kernel(0, np.log(1), MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[np.zeros(6), np.ones(2)], dtype=np.bool)))
        #noise_kernel = create_noise_kernel(0, np.log(1), MaskedFeatureKernel(ARDSEKernel(np.log(l[6:7]), np.log(1)), np.array(np.r_[np.zeros(6), np.ones(2)], dtype=np.bool)))
        #kernel = MaskedFeatureKernel(kernel, fmask) + noise_kernel
        
        #mtl kernel
        #noise_kernel = NoiseKernel(np.log(0.5)) #+ TaskNoiseKernel(X[train,0], 0, np.log(0.001))
        #mtl_kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[0, np.ones(5), np.zeros(2)] ,dtype=bool))*MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[0, np.zeros(5), np.ones(2)] ,dtype=bool))
        #mtl_kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[0, np.ones(5), np.zeros(2)] ,dtype=bool))*MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[0, np.ones(7)] ,dtype=bool))
        #mtl_kernel = mtl_kernel + MaskedFeatureKernel(SqConstantKernel(np.log(1)) * LinearKernel(), fmask)
        #kernel = FixedParameterKernel(mtl_kernel+noise_kernel, [3])
        
        #algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=False)
        #model = GPRegression(kernel, infer_method=ExactInference)
        #algo = PITCSparseGPRegression(igroup, kernel, noise_kernel, infer_method=PITCExactInference, selector=selector, fix_inducing=False)
        #algo = GPRegression(kernel, meanfct=meanfct, infer_method=ExactInference)
        weights[i] = len(test)
        
        mse[i], nmse[i], mll[i], nmll[i], yfit, var = eval_regression(data_train, data_test, task_ids, model)
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
