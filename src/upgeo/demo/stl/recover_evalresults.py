'''
Created on Oct 22, 2012

@author: marcel
'''
import numpy as np
import upgeo.util.stats as stats
import upgeo.util.metric as metric

from upgeo.demo.util import loadmat_data, loadmat_folds
from numpy.core.numeric import array_str

if __name__ == '__main__':
    nfolds = 10
    fold_filename = '//home/marcel/datasets/multilevel/eusinan/bssa/splitz2/eu_{0}_indexes.mat'    
    #filename = '/home/marcel/datasets/multilevel/eusinan/bssa/eval_eudata_norm2.mat'
    filename = '/home/marcel/datasets/multilevel/eusinan/bssa/eval_eudata.mat'
    
    
    X,Y = loadmat_data(filename)

    n = X.shape[0]
    Ypred = np.zeros((n,2)) #matrix of the fitted value and its variance

    #choose target variable (see readme file fot the index coding)
    Y = Y[:,9]

    #Ypred = np.loadtxt('/home/marcel/datasets/multilevel/eusinan/bssa/results/pgv/splitz1/gpselin_data3', delimiter=',')
    Ypred = np.loadtxt('/home/marcel/datasets/multilevel/eusinan/bssa/results/nico_depth/testerror_depth_re_T4_splitz2.csv', delimiter=',')
    
    mse = np.empty(nfolds)
    mll = np.empty(nfolds)
    nmse = np.empty(nfolds)
    nmll = np.empty(nfolds)
    weights = np.empty(nfolds)
    
    
    
    for i in xrange(nfolds):
        train, test = loadmat_folds(fold_filename.format(i+1))
        
        
        yfit = Ypred[test,0] 
        var = Ypred[test,1]
        
        weights[i] = len(test) 
        mse[i] = metric.mspe(Y[test], yfit)
        nmse[i] = mse[i]/np.var(Y[test])
        mll[i] = metric.nlp(Y[test], yfit, var)
        nmll[i] = mll[i]-metric.nlpp(Y[test], np.mean(Y[train]), np.var(Y[train]))
        
    
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