'''
Created on Apr 29, 2013

@author: marcel
'''
import numpy as np
import upgeo.util.metric as metric
import upgeo.util.stats as stats

from upgeo.mtl.gp import CMOGPRegression
from upgeo.mtl.infer import CMOGPExactInference
from upgeo.base.kernel import FixedParameterKernel, SEKernel, LinearKernel,\
    SqConstantKernel, DiracConvolvedKernel, ARDSEKernel, MaskedFeatureKernel,\
    GroupNoiseKernel, NoiseKernel
from upgeo.demo.util import loadmat_folds, loadmat_data_and_periods
from upgeo.util.array import unique
from numpy.core.numeric import array_str
from upgeo.base.selector import RandomSubsetSelector
from upgeo.mtl.kernel import ConvolvedMTLKernel

def eval_cmogp(train, test, itasks, task_idx, kernel):
    algo = CMOGPRegression(kernel, infer_method=CMOGPExactInference) 
    
    Xtrain = train[0]
    Ytrain = train[1]

    Xtest = test[0]
    Ytest = test[1]
    
    
    
    ntasks = len(itasks)
    n,d = Xtest.shape
    
    mask = np.ones(d, dtype=bool)
    mask[task_idx] = 0
    
    algo.fit(Xtrain, Ytrain, itasks)
    
    yfit = np.empty(n)
    var = np.empty(n)
    #this kind of prediction does only work 
    for i in xrange(ntasks):
        idxs = Xtest[:,task_idx] == i
        yfit[idxs], var[idxs] = algo.predict_task(Xtest[idxs,:][:,mask], i, ret_var=True)
        
    
    mse = metric.mspe(Ytest, yfit)
    nmse = mse/np.var(Ytest)
    mll = metric.nlp(Ytest, yfit, var)
    nmll = mll-metric.nlpp(Ytest, np.mean(Ytrain), np.var(Ytrain))
     
    return mse, nmse, mll, nmll, yfit, var

def prepare_train_data(X, Y, task_idx):
    tasks = X[:, task_idx]
    utasks = np.sort(np.unique(tasks))
    
    d = X.shape[1]
    mask = np.ones(d, dtype=bool)
    mask[task_idx] = 0
    
    Xmtl = np.empty((0,X.shape[1]-1))
    Ymtl = np.empty((0))
    ntasks = len(utasks)
    itasks = np.zeros(ntasks)
    
    for i in xrange(ntasks): 
        t = utasks[i]
        print Y[tasks==t].shape
        Xmtl = np.r_[Xmtl, X[tasks==t,:][:,mask]]
        Ymtl = np.r_[Ymtl, Y[tasks==t]]
        if i != ntasks-1:
            itasks[i+1] = itasks[i] + np.sum(tasks==t)

    return Xmtl, Ymtl, itasks


def create_mtl_kernel(ntasks, nfeatures, l=None, fmask=None, grp_idx=None):
    #construct mtl kernel
    
    if fmask == None:
        
        #lKernel = FixedParameterKernel(SEKernel(np.log(1),np.log(1)), [1])
        
        #latent_kernel = DiracConvolvedKernel(FixedParameterKernel(SEKernel(np.log(0.1),np.log(1)), [1]))
        lKernel = FixedParameterKernel(SEKernel(np.log(0.1),np.log(1)), [1])
        #lKernel = SEKernel(np.log(1),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel()
        #lKernel = FixedParameterKernel(ARDSEKernel(np.log(l),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel(), [len(l)])
        #lKernel = FixedParameterKernel(ARDSEKernel(np.log(l),np.log(1)), [len(l)])
        dpKernel = DiracConvolvedKernel(lKernel)
        idpKernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #idpKernel = ARDSEKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    else:
        lKernel = FixedParameterKernel(MaskedFeatureKernel(SEKernel(np.log(0.1),np.log(1)), fmask), [1])
        #lKernel = FixedParameterKernel(MaskedFeatureKernel(SEKernel(np.log(1),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel(), fmask), [1])
        #lKernel = SEKernel(np.log(1),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel()
        #lKernel = FixedParameterKernel(MaskedFeatureKernel(ARDSEKernel(np.log(l),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel(), fmask), [len(l)])
        #lKernel = FixedParameterKernel(MaskedFeatureKernel(ARDSEKernel(np.log(l),np.log(1)), fmask), [len(l)])
    
        dpKernel = DiracConvolvedKernel(lKernel)
        #idpKernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask)
        idpKernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask) + NoiseKernel(np.log(0.5))
        #idpKernel = MaskedFeatureKernel(ARDSEKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask)+ NoiseKernel(np.log(0.5))
    
    if grp_idx != None:
        idpKernel = idpKernel + GroupNoiseKernel(grp_idx, np.log(1))
        #idpKernel = idpKernel + CorrelatedNoiseKernel(grp_idx, np.log(0.1), np.log(0.5))
    
    #theta = np.r_[np.log(0.1)*np.ones(nfeatures), np.log(1)]
    theta = [np.log(1)]
    kernel = ConvolvedMTLKernel(dpKernel, theta, ntasks, idpKernel)
    return kernel   



if __name__ == '__main__':
    nfolds = 5
    fold_filename = '//home/marcel/datasets/multilevel/nga/ssa/splitz/nga_{0}_indexes.mat'
    
    filename = '/home/marcel/datasets/multilevel/nga/ssa/ssa_eval_mech_eq.mat'
    
    X,Y,periods = loadmat_data_and_periods(filename)
    #for simple pitc approximation
    eqs = X[:,0]
    #X = X[:,1:]
    #print Y
    #choose period of the response spectra as target variable
    #print periods == 0.0
    period_idx = np.flatnonzero(periods == 0)
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
    
    task_idx = 2
    
    for i in xrange(nfolds):
        train, test = loadmat_folds(fold_filename.format(i+1))
        mvs_train = np.isnan(y[train])
        mvs_test = np.isnan(y[test])
    
    
        Xtrain, ytrain, itasks = prepare_train_data(X[~mvs_train], y[train[~mvs_train]], task_idx)
    
        data_train = (Xtrain, ytrain)
        data_test = (X[test[~mvs_test]], y[test[~mvs_test]])
        
        _, igroup = unique(eqs[~mvs_train], True)
        
        #mask for selecting base features for the cov function 
        #fmask = np.r_[1, 0, np.ones(5), 0, 0, 1] #distance  (data2 features)
        #fmask = np.r_[1, 0, np.ones(4), 0, 1, 0, 1] #log distance (data3 features)
        fmask = np.r_[0, np.ones(4)]
        fmask = np.array(fmask, dtype=np.bool)
        
        #init lenght-scales
        #l = (np.max(data_train[0],0)-np.min(data_train[0],0))/2
        l = (np.max(data_train[0][:,fmask],0)-np.min(data_train[0][:,fmask],0))/2
        l[l == 0] = 1e-4
        #print 'l={0}'.format(l)
        
        ntasks = len(itasks)
        nfeatures = Xtrain.shape[1]
        kernel = create_mtl_kernel(ntasks, nfeatures, l, fmask, 0)
        
        weights[i] = len(test)
        
        mse[i], nmse[i], mll[i], nmll[i], yfit, var = eval_cmogp(data_train, data_test, itasks, task_idx, kernel)
        
        ypred[test[~mvs_test],0] = yfit
        ypred[test[~mvs_test],1] = var
        
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
        
    #np.savetxt('/home/mhermkes/datasets/multilevel/nga/ssa/results/pga/gpsepoly2_grpnoise_data2', ypred, delimiter=',')
