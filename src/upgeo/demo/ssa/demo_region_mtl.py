'''
Created on Mar 21, 2013

@author: marcel
'''
import numpy as np
import upgeo.util.metric as metric
import upgeo.util.stats as stats


from upgeo.util.array import unique
from upgeo.demo.util import loadmat_folds, loadmat_mtl_data
from upgeo.base.selector import RandomSubsetSelector, FixedSelector,\
    KMeansSelector
from numpy.core.numeric import array_str
from upgeo.regression.bayes import EMBayesRegression
from upgeo.base.kernel import ExpGaussianKernel, SEKernel, SqConstantKernel,\
    NoiseKernel, LinearKernel, FixedParameterKernel, DiracConvolvedKernel,\
    ARDSEKernel, ExpARDGaussianKernel, CompoundKernel, ARDLinearKernel,\
    GroupNoiseKernel, HiddenKernel, MaskedFeatureKernel,\
    MaskedFeatureConvolvedKernel
from upgeo.mtl.kernel import ConvolvedMTLKernel
from upgeo.mtl.gp import SparseCMOGPRegression, STLGPRegression,\
    PooledGPRegression
from upgeo.mtl.infer import SparseCMOGPExactInference
from upgeo.util.glob import APPROX_TYPE
from upgeo.base.infer import ExactInference

def eval_stl_linreg(train, test, task_ids):
    Xtrain = train[0]
    Ytrain = train[1]
    Gtrain = train[2]
    
    Xtest = test[0]
    Ytest = test[1]
    Gtest = test[2]
    
    k = len(task_ids)
    
    mse = np.empty(k)
    nmse = np.empty(k)
    mll = np.empty(k)
    nmll = np.empty(k)
    
    Yfit = np.zeros(n)
    Var = np.zeros(n)
    
    for i in xrange(k):
        #norm_period = (periods[i]-min_periop)/(max_period-min_period)
        #m = np.sum(~Ytest_nan[:,i])
        
        train_ids = Gtrain == task_ids[i]
        test_ids = Gtest == task_ids[i]
        
        algo = EMBayesRegression(alpha0=1, beta0=1, weight_bias=True)
        algo.fit(Xtrain[train_ids], Ytrain[train_ids])
        yfit, var = algo.predict(Xtest[test_ids], ret_var=True)
     
        Yfit[test_ids] = yfit
        Var[test_ids] = var
        
        print 'sum_testdata={0}'.format(np.sum(test_ids))
        
        mse[i] = metric.mspe(Ytest[test_ids], yfit)
        nmse[i] = mse[i]/np.var(Ytest[test_ids])
        
        mll[i] = metric.nlp(Ytest[test_ids], yfit, var)
        nmll[i] = mll[i]-metric.nlpp(Ytest[test_ids], np.mean(Ytrain[train_ids]), np.var(Ytrain[train_ids]))
     
    return mse, nmse, mll, nmll, Yfit, Var

def eval_pooled_linreg(train, test, task_ids):
    Xtrain = train[0]
    Ytrain = train[1]
    Gtrain = train[2]
    
    Xtest = test[0]
    Ytest = test[1]
    Gtest = test[2]
    
    k = len(task_ids)
    
    mse = np.empty(k)
    nmse = np.empty(k)
    mll = np.empty(k)
    nmll = np.empty(k)
    
    Yfit = np.zeros(n)
    Var = np.zeros(n)
    
    algo = EMBayesRegression(alpha0=1, beta0=1, weight_bias=True)
    algo.fit(Xtrain, Ytrain)
    
    for i in xrange(k):
        #norm_period = (periods[i]-min_periop)/(max_period-min_period)
        #m = np.sum(~Ytest_nan[:,i])
        
        train_ids = Gtrain == task_ids[i]
        test_ids = Gtest == task_ids[i]
        
        yfit, var = algo.predict(Xtest[test_ids], ret_var=True)
     
        Yfit[test_ids] = yfit
        Var[test_ids] = var
        
        mse[i] = metric.mspe(Ytest[test_ids], yfit)
        nmse[i] = mse[i]/np.var(Ytest[test_ids])
        
        mll[i] = metric.nlp(Ytest[test_ids], yfit, var)
        nmll[i] = mll[i]-metric.nlpp(Ytest[test_ids], np.mean(Ytrain[train_ids]), np.var(Ytrain[train_ids]))
     
    return mse, nmse, mll, nmll, Yfit, Var


def eval_mtl_gp(train, test, task_ids, gp):
    Xtrain = train[0]
    Ytrain = train[1]
    Gtrain = train[2]
    _,itask = unique(Gtrain,True)
    
    Xtest = test[0]
    Ytest = test[1]
    Gtest = test[2]
    
    k = len(task_ids)
    
    mse = np.zeros(k)
    nmse = np.zeros(k)
    mll = np.zeros(k)
    nmll = np.zeros(k)
    
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
        
        mse[i] = metric.mspe(Ytest[test_ids], yfit)
        nmse[i] = mse[i]/np.var(Ytest[test_ids])
        
        mll[i] = metric.nlp(Ytest[test_ids], yfit, var)
        nmll[i] = mll[i]-metric.nlpp(Ytest[test_ids], np.mean(Ytrain[train_ids]), np.var(Ytrain[train_ids]))
     
    return mse, nmse, mll, nmll, Yfit, Var
    
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
    fold_filename = '//home/marcel/datasets/multilevel/nga/ssa/transfer/euregbig_splitz/nga_{0}_indexes.mat'
    
    filename = '/home/marcel/datasets/multilevel/nga/ssa/transfer/mtl_eudata_big_eq.mat'
    
    #assuming that data is sorted by the tasks
    X,y,tasks = loadmat_mtl_data(filename)
    task_ids = unique(tasks)
    k = len(task_ids)
    #Y = Y[:,0:5]
    #periods = periods[0:5]
    #print Y
    #choose period of the response spectra as target variable
    #print periods == 0.0
    #period_idx = np.flatnonzero(periods == 8)
    #print period_idx
    #y = np.squeeze(Y[:,period_idx])
    
    n = X.shape[0]
     
    
    Yfit = np.zeros((y.shape[0]))
    Var = np.zeros((y.shape[0]))
        
    mse = np.empty((nfolds,k))
    mll = np.empty((nfolds,k))
    nmse = np.empty((nfolds,k))
    nmll = np.empty((nfolds,k))
    weights = np.empty(nfolds)
    
    for i in xrange(nfolds):
        train, test = loadmat_folds(fold_filename.format(i+1))
        
        data_train = (X[train], y[train], tasks[train])
        data_test = (X[test], y[test], tasks[test])
        
        #mask for selecting base features for the cov function 
        #fmask = np.r_[1, 0, np.ones(5), 0, 0, 1] #distance  (data2 features)
        #fmask = np.r_[1, 0, np.ones(4), 0, 1, 0, 1] #log distance (data3 features)
        fmask = np.r_[0, np.ones(7)]
        fmask = np.array(fmask, dtype=np.bool)
        
        #init lenght-scales
        l = (np.max(data_train[0],0)-np.min(data_train[0],0))/2
        #l = (np.max(data_train[0][:,fmask],0)-np.min(data_train[0][:,fmask],0))/2
        l[l == 0] = 1e-4
        print 'l={0}'.format(l)
        
        #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() #+ NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(0.001)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + PolynomialKernel(2, np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) #+ NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5)) + PolynomialKernel(2, np.log(1), np.log(1))
        #kernel = RBFKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = RBFKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(1)*np.ones(len(l)), np.log(1))# + NoiseKernel(np.log(0.5))
        kernel = ARDSEKernel(np.log(1)*np.ones(len(l)), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() #+ NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(1)*np.ones(len(l)), np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(len(l)), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(l), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(l), np.log(1)) + ARDPolynomialKernel(2, np.log(1)*np.ones(len(l)), np.log(1), np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(len(l)), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDSELinKernel(np.log(l), np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDRBFKernel(np.log(l), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = ARDRBFKernel(np.log(l), np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(len(l)), np.log(1))  + NoiseKernel(np.log(0.5))

        #create complex noise model
        noise_kernel = create_noise_kernel(0, np.log(1)) + NoiseKernel(np.log(0.5))
        #noise_kernel = create_noise_kernel(0, np.log(1), MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[np.zeros(6), np.ones(2)], dtype=np.bool)))
        #noise_kernel = create_noise_kernel(0, np.log(1), MaskedFeatureKernel(ARDSEKernel(np.log(l[6:7]), np.log(1)), np.array(np.r_[np.zeros(6), np.ones(2)], dtype=np.bool)))
        kernel = MaskedFeatureKernel(kernel, fmask) + noise_kernel

        
        #gp = STLGPRegression(kernel, infer_method=ExactInference)
        #gp = PooledGPRegression(kernel, infer_method=ExactInference)
        
        #selector = RandomSubsetSelector(15)
        selector = KMeansSelector(30, False) 
        Xu = selector.apply(data_train[0], data_train[1])
        selector = FixedSelector(Xu)
        #
        
        
        #latent_kernel = ExpGaussianKernel(np.log(0.1))
        latent_kernel = ExpARDGaussianKernel(np.ones(7)*np.log(0.1))
        #latent_kernel = CompoundKernel([ExpGaussianKernel(np.log(0.1)), ExpGaussianKernel(np.log(0.2))])
        #latent_kernel = DiracConvolvedKernel(FixedParameterKernel(SEKernel(np.log(0.1),np.log(1)), [1]))       
        #latent_kernel = DiracConvolvedKernel(FixedParameterKernel(SEKernel(np.log(0.01),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel(), [1]))
        #latent_kernel = DiracConvolvedKernel(FixedParameterKernel(ARDSEKernel(np.ones(7)*np.log(0.1),np.log(1)), [7]))
        #latent_kernel = CompoundKernel([DiracConvolvedKernel(FixedParameterKernel(ARDSEKernel(np.ones(7)*np.log(0.1),np.log(1)), [7])), DiracConvolvedKernel(FixedParameterKernel(ARDSEKernel(np.ones(7)*np.log(0.25),np.log(1)), [7]))])
        #latent_kernel = CompoundKernel([ExpARDGaussianKernel(np.ones(7)*np.log(0.1)), ExpARDGaussianKernel(np.log(np.random.random(7)+0.0001))])
        #latent_kernel = CompoundKernel([ExpARDGaussianKernel(np.ones(7)*np.log(0.1)), ExpARDGaussianKernel(np.ones(7)*np.log(0.2))])
        #latent_Kernel = DiracConvolvedKernel(GaussianKernel(np.log(1)))
        #noise_kernel = SEKernel(np.log(0.1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #noise_kernel = ARDSEKernel(np.ones(7)*np.log(0.1),np.log(1))+ SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #noise_kernel = ARDSEKernel(np.ones(7)*np.log(0.1),np.log(1))+ NoiseKernel(np.log(0.3))
        noise_kernel = SEKernel(np.log(0.1), np.log(1)) + NoiseKernel(np.log(0.3))
        #noise_kernel = TaskNoiseKernel((periods-np.min(periods))/(np.max(periods)-np.min(periods)), 7, np.log(0.5))
        #noise_kernel = TaskNoiseKernel((periods-np.min(periods))/(np.max(periods)-np.min(periods)), 7, np.log(0.5))
        
        
        noise_kernel = MaskedFeatureKernel(noise_kernel, fmask) + create_noise_kernel(0, np.log(1)) + NoiseKernel(np.log(0.5))
        latent_kernel = MaskedFeatureConvolvedKernel(latent_kernel, fmask)
        #theta = [np.log(0.1), np.log(1)]
        #theta = [np.log(0.1), np.log(1), np.log(0.2), np.log(1)]
        theta = np.r_[np.ones(7)*np.log(0.1), np.log(1)]
        #theta = np.r_[np.ones(7)*np.log(0.1), np.log(1), np.ones(7)*np.log(0.2), np.log(1)]
        #theta = [np.log(1)]
        #theta = [np.log(1),np.log(1)]
        #theta = [np.log(1), np.log(1)]
        #theta = [np.log(0.01), np.log(1)]   
        kernel = ConvolvedMTLKernel(latent_kernel, theta, k, noise_kernel) 
        idx = [7,15]
        #kernel._theta[:,idx] = np.log(np.random.rand(k,len(idx)))   
        
        #gp = SparseCMOGPRegression(kernel, beta=100, infer_method=SparseCMOGPExactInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=True)        
        gp = SparseCMOGPRegression(kernel, infer_method=SparseCMOGPExactInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=True)
        
        #algo = SparseCMOGPRegression(kernel, infer_method=SparseCMOGPOnePassInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=False)
        
        #algo = SparseGPRegression(kernel, infer_method=FITCExactInference, selector=selector, fix_inducing=True)
        #algo = GPRegression(kernel, infer_method=ExactInference)
        #algo = GPRegression(kernel, meanfct=meanfct, infer_method=ExactInference)
        weights[i] = len(test)
        #mse[i], nmse[i], mll[i], nmll[i], Yfit[test], Var[test] = eval_stl_linreg(data_train, data_test, task_ids)
        #mse[i], nmse[i], mll[i], nmll[i], Yfit[test], Var[test] = eval_pooled_linreg(data_train, data_test, task_ids)
        mse[i], nmse[i], mll[i], nmll[i], Yfit[test], Var[test] = eval_mtl_gp(data_train, data_test, task_ids, gp)
        #mse[i], nmse[i], mll[i], nmll[i], Yfit[test], Var[test] = eval_reg(data_train, data_test, algo, periods)
        #mse[i], nmse[i], mll[i], nmll[i], Yfit[test], Var[test] = eval_optsparsegp(data_train, data_test, periods, kernel, noise_kernel, selector)
        #mse[i], nmse[i], mll[i], nmll[i], Yfit[test], Var[test] = eval_mogp(data_train, data_test, algo, period_idx)
        
        
        #print 'hyperparams={0}'.format(kernel.        params)
        print 'task({0}): mse={1},{2}, mll={3},{4}'.format(i, mse[i], nmse[i], mll[i], nmll[i])
        
    for i in xrange(k):
        print 'CV Results for region={0}:'.format(task_ids[i])
        print 'mse'     
        print array_str(mse[:,i], precision=16)
        print 'nmse'
        print array_str(nmse[:,i], precision=16)
        print 'mll'
        print array_str(mll[:,i], precision=16)
        print 'nmll'
        print array_str(nmll[:,i], precision=16)
        print 'Total Results:'
                
        if task_ids[i]==8:
            #mask = np.array([0,1,2,4])
            mask = np.array([0,2,3])
            print 'Output Result:{0}'.format(i) 
            means = np.asarray([stats.mean(mse[mask,i], weights[mask]), stats.mean(nmse[mask,i], weights[mask]), 
                                stats.mean(mll[mask,i], weights[mask]), stats.mean(nmll[mask,i], weights[mask])])
            std = np.asarray([stats.stddev(mse[mask,i], weights[mask]), stats.stddev(nmse[mask,i], weights[mask]),
                              stats.stddev(mll[mask,i], weights[mask]), stats.stddev(nmll[mask,i], weights[mask])])
                
            print 'mean={0}'.format(array_str(means, precision=16))
            print 'err={0}'.format(array_str(std, precision=16))
    
        else:
            
            print 'Output Result:{0}'.format(i) 
            means = np.asarray([stats.mean(mse[:,i], weights), stats.mean(nmse[:,i], weights), 
                                stats.mean(mll[:,i], weights), stats.mean(nmll[:,i], weights)])
            std = np.asarray([stats.stddev(mse[:,i], weights), stats.stddev(nmse[:,i], weights),
                              stats.stddev(mll[:,i], weights), stats.stddev(nmll[:,i], weights)])
                
            print 'mean={0}'.format(array_str(means, precision=16))
            print 'err={0}'.format(array_str(std, precision=16))
        
    #np.savetxt('/home/mhermkes/datasets/multilevel/nga/ssa/results/pga/gpsepoly2_grpnoise_data2', ypred, delimiter=',')
