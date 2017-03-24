'''
Created on Oct 17, 2012

@author: marcel
'''
import numpy as np
import upgeo.util.metric as metric
import upgeo.util.stats as stats

from upgeo.demo.util import loadmat_data, loadmat_folds
from upgeo.base.kernel import SEKernel, NoiseKernel, SqConstantKernel,\
    LinearKernel, ARDSEKernel, RBFKernel
from upgeo.base.selector import KMeansSelector
from upgeo.base.gp import GPRegression, SparseGPRegression
from upgeo.base.infer import ExactInference, FITCExactInference
from upgeo.regression.bayes import EMBayesRegression, EBChenRegression
from numpy.core.numeric import array_str
from upgeo.regression.linear import LSRegresion

def eval_reg(train, test, algo):
    Xtrain = train[0]
    Ytrain = train[1]

    Xtest = test[0]
    Ytest = test[1]
 
    algo.fit(Xtrain, Ytrain)
    yfit, var = algo.predict(Xtest, ret_var=True)
    
    mse = metric.mspe(Ytest, yfit)
    nmse = mse/np.var(Ytest)
    mll = metric.nlp(Ytest, yfit, var)
    nmll = mll-metric.nlpp(Ytest, np.mean(Ytrain), np.var(Ytrain))
     
    return mse, nmse, mll, nmll, yfit, var


if __name__ == '__main__':
    nfolds = 10
    fold_filename = '//home/marcel/datasets/multilevel/eusinan/bssa/splitz2/eu_{0}_indexes.mat'
    
    filename = '/home/marcel/datasets/multilevel/eusinan/bssa/eval_eudata_norm6.mat'
    
    X,Y = loadmat_data(filename)

    n = X.shape[0]
    Ypred = np.zeros((n,2)) #matrix of the fitted value and its variance

    #choose target variable (see readme file fot the index coding)
    Y = Y[:,9]
    
    mse = np.empty(nfolds)
    mll = np.empty(nfolds)
    nmse = np.empty(nfolds)
    nmll = np.empty(nfolds)
    weights = np.empty(nfolds)
    
    
    
    for i in xrange(nfolds):
        #train, test = load_folds(fold_filename.format(j+1,i+1))
        train, test = loadmat_folds(fold_filename.format(i+1))
        
        data_train = (X[train], Y[train])
        data_test = (X[test], Y[test])
        
    
        l = (np.max(data_train[0],0)-np.min(data_train[0],0))/2
        l[l == 0] = 1e-4
        print 'l={0}'.format(l)
        
        algo = EMBayesRegression(alpha0=1, beta0=1, weight_bias=True)
        #algo = EBChenRegression(alpha0=1, beta0=1, weight_bias=True)
        #algo = LSRegresion()
        #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = RBFKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    
        #selector = KMeansSelector(30, False) 
        
        #algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=False)
        #algo = GPRegression(kernel, infer_method=ExactInference)
        weights[i] = len(test)
        
        mse[i], nmse[i], mll[i], nmll[i], yfit, var = eval_reg(data_train, data_test, algo)
        Ypred[test,0] = yfit
        Ypred[test,1] = var
        
        #mse[i], r2[i] = eval_multiple_gp(data_train, data_test, kernels, None, False, True)
        #mse[i], r2[i] = eval_multiple_sgp(data_train, data_test, kernels, 15, None, False, True)
        #mse[i], r2[i] = eval_gp(data_train, data_test, kernel, meanfct, False, True)
        #mse[i], r2[i] = eval_bhcsgp(data_train, data_test, kernel, False)
        #mse[i], r2[i] = eval_bhcrobustsgp(data_train, data_test, kernel, False)
        #mse[i], r2[i] = eval_bhclinreg(data_train, data_test, False)
        #mse[i], r2[i] = eval_bhcrobustreg(data_train, data_test, False)
        
        
        print 'hyperparams={0}'.format(kernel.params)
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
        
    np.savetxt('/home/marcel/datasets/multilevel/eusinan/bssa/results/T4/splitz2/linreg_data6', Ypred, delimiter=',')

