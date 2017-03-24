'''
Created on Oct 1, 2012

@author: marcel
'''
import numpy as np
import upgeo.util.metric as metric
import upgeo.util.stats as stats

from upgeo.demo.util import loadmat_data, loadmat_folds, prepare_mtl_data
from upgeo.base.kernel import ExpGaussianKernel, ARDSEKernel, NoiseKernel,\
    SqConstantKernel, LinearKernel
from upgeo.mtl.kernel import ConvolvedMTLKernel
from upgeo.mtl.gp import CMOGPRegression
from numpy.core.numeric import array_str
from upgeo.mtl.infer import CMOGPOnePassInference, CMOGPExactInference


def eval_cmogp(train, test, itasks, kernel):
    algo = CMOGPRegression(kernel, infer_method=CMOGPExactInference) 
    
    Xtrain = train[0]
    Ytrain = train[1]

    Xtest = test[0]
    Ytest = test[1]
    
    ntasks = len(itasks)
    n = Ytest.shape[0]
    
    algo.fit(Xtrain, Ytrain, itasks)
    yfit, var = algo.predict(Xtest, ret_var=True)
    
    smse = np.zeros(ntasks)
    msll = np.zeros(ntasks)
    itasks = np.r_[itasks, len(Ytrain)]
    for i in xrange(ntasks):
        start = itasks[i]
        end = itasks[i+1]
        smse[i] = metric.mspe(Ytest[:,i], yfit[:,i])/np.var(Ytest[:,i])
        mll = np.sum((yfit[:,i]-Ytest[:,i])**2/(2*var[:,i]) + 0.5*np.log(2*np.pi*var[:,i]))/n
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

def create_mtl_kernel(ntasks, nfeatures):
    #construct mtl kernel
    dpKernel = ExpGaussianKernel(np.log(0.1)*np.ones(nfeatures))
    #dpKernel = DiracConvolvedKernel(SEKernel(np.log(1),np.log(1)))
    idpKernel = ARDSEKernel(np.log(1)*np.ones(nfeatures), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(np.sqrt(0.135335283236613)))
    theta = np.r_[np.log(0.1)*np.ones(nfeatures), np.log(1)]
    #theta = [np.log(1)]
    kernel = ConvolvedMTLKernel(dpKernel, theta, ntasks, idpKernel)
    return kernel

if __name__ == '__main__':
    nsplitz = 1
    nfolds = 10
    fold_filename = '//home/marcel/datasets/multilevel/nga/pooled/mainshock/splitz30dames/nga_{0}_indexes.mat'
    
    filename = '/home/marcel/datasets/multilevel/nga/pooled/mainshock/mtleval_nga_logD30.mat'
    
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
        
        
    
        kernel = create_mtl_kernel(ntasks, nfeatures)
        #mse[i], r2[i] = eval_linreg(data_train, data_test, False, True)
        weights[i] = len(test)
        
        smse[i], msll[i] = eval_cmogp(data_train, data_test, itasks, kernel)
        print 'hyperparams={0}'.format(kernel.params)
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
        
    

