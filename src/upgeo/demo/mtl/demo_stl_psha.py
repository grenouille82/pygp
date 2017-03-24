'''
Created on Oct 1, 2012

@author: marcel
'''
from upgeo.regression.bayes import EMBayesRegression
from upgeo.base.gp import SparseGPRegression, GPRegression
from upgeo.base.selector import KMeansSelector
from upgeo.base.infer import FITCExactInference, ExactInference
'''
Created on Oct 1, 2012

@author: marcel
'''
import numpy as np
import upgeo.util.metric as metric
import upgeo.util.stats as stats

from upgeo.demo.util import loadmat_data, loadmat_folds, prepare_mtl_data
from upgeo.base.kernel import ExpGaussianKernel, ARDSEKernel, NoiseKernel,\
    SqConstantKernel, LinearKernel, SEKernel
from upgeo.mtl.kernel import ConvolvedMTLKernel
from upgeo.mtl.gp import CMOGPRegression
from numpy.core.numeric import array_str
from upgeo.mtl.infer import CMOGPOnePassInference


def eval_stlalgo(train, test, itasks, algo):
    Xtrain = train[0]
    Ytrain = train[1]

    Xtest = test[0]
    Ytest = test[1]
    
    ntasks = len(itasks)
    n = Ytest.shape[0]
     
    params = np.copy(algo.hyperparams)
     
    smse = np.zeros(ntasks)
    msll = np.zeros(ntasks)
    itasks = np.r_[itasks, len(Ytrain)]
    for i in xrange(ntasks):
        start = itasks[i]
        end = itasks[i+1]
        
        algo.hyperparams = params
        algo.fit(Xtrain[start:end], Ytrain[start:end])
        yfit, var = algo.predict(Xtest, ret_var=True)
        
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

if __name__ == '__main__':
    nfolds = 10
    fold_filename = '//home/marcel/datasets/multilevel/nga/pooled/mainshock/splitz30dames/nga_{0}_indexes.mat'
    
    filename = '/home/marcel/datasets/multilevel/nga/pooled/mainshock/mtleval_nga_D30.mat'
    
    X,Y = loadmat_data(filename)
    nfeatures = X.shape[1]
    ntasks = Y.shape[1]
    
    smse = np.empty((nfolds,ntasks))
    msll = np.empty((nfolds,ntasks))
    weights = np.empty(nfolds)

    for i in xrange(nfolds):
        #train, test = load_folds(fold_filename.format(j+1,i+1))
        train, test = loadmat_folds(fold_filename.format(i+1))
    
        Xtrain, ytrain, itasks = prepare_mtl_data(X[train], Y[train])
    
        data_train = (Xtrain, ytrain)
        data_test = (X[test], Y[test])
        
        
    
        l = (np.max(data_train[0],0)-np.min(data_train[0],0))/2
        l[l == 0] = 1e-4
        print 'l={0}'.format(l)
        
        #algo = EMBayesRegression(alpha0=1, beta0=1, weight_bias=True)
        #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(1)*np.ones(nfeatures), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(1))
        selector = KMeansSelector(30, False) 
        
        #algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=False)
        algo = GPRegression(kernel, infer_method=ExactInference)
        weights[i] = len(test)
        
        smse[i], msll[i] = eval_stlalgo(data_train, data_test, itasks, algo)
        
        #mse[i], r2[i] = eval_multiple_gp(data_train, data_test, kernels, None, False, True)
        #mse[i], r2[i] = eval_multiple_sgp(data_train, data_test, kernels, 15, None, False, True)
        #mse[i], r2[i] = eval_gp(data_train, data_test, kernel, meanfct, False, True)
        #mse[i], r2[i] = eval_bhcsgp(data_train, data_test, kernel, False)
        #mse[i], r2[i] = eval_bhcrobustsgp(data_train, data_test, kernel, False)
        #mse[i], r2[i] = eval_bhclinreg(data_train, data_test, False)
        #mse[i], r2[i] = eval_bhcrobustreg(data_train, data_test, False)
        
        
        #print 'hyperparams={0},{1}'.format(kernel.params, meanfct.params)
        print 'task({0}): smse={1}, msll={2}'.format(i, smse[i], msll[i])
        
    print 'CV Results:'
    print 'smse'
    print array_str(smse, precision=16)
    print 'msll'
    print array_str(msll, precision=16)
    print 'Total Results:'
    
    for i in xrange(ntasks):
        print 'Output Result:{0}'.format(i) 
        means = np.asarray([stats.mean(smse[:,i], weights), stats.mean(msll[:,i], weights)])
        std = np.asarray([stats.stddev(smse[:,i], weights), stats.stddev(msll[:,i], weights)])
        
        print 'mean={0}'.format(array_str(means, precision=16))
        print 'err={0}'.format(array_str(std, precision=16))
        
    

