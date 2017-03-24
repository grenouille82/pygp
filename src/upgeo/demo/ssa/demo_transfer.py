'''
Created on Mar 20, 2013

@author: marcel
'''

import numpy as np
import upgeo.util.metric as metric
import upgeo.util.stats as stats

from upgeo.demo.util import loadmat_folds, loadmat_transfer_data
from numpy.core.numeric import array_str
from upgeo.regression.bayes import EMBayesRegression
from upgeo.mtl.infer import SparseCMOGPOnePassInference,\
    SparseCMOGPExactInference
from upgeo.util.glob import APPROX_TYPE
from upgeo.base.selector import RandomSubsetSelector, FixedSelector,\
    KMeansSelector
from upgeo.mtl.gp import SparseCMOGPRegression
from upgeo.base.kernel import LinearKernel, NoiseKernel, SqConstantKernel,\
    SEKernel, ARDSEKernel, ARDLinearKernel, GroupNoiseKernel, HiddenKernel,\
    MaskedFeatureKernel, ExpGaussianKernel
from upgeo.base.gp import GPRegression
from upgeo.base.infer import ExactInference
from upgeo.mtl.kernel import ConvolvedMTLKernel

def eval_stl_transfer_reg(train, test, background, algo):
    Xtrain = train[0]
    Ytrain = train[1]

    Xtest = test[0]
    Ytest = test[1]
 
    algo.fit(Xtrain, Ytrain)
    yfit, var = algo.predict(Xtest, ret_var=True)
    print 'hyperparams={0}'.format(np.exp(algo.hyperparams))
    
    mse = metric.mspe(Ytest, yfit)
    nmse = mse/np.var(Ytest)
    mll = metric.nlp(Ytest, yfit, var)
    nmll = mll-metric.nlpp(Ytest, np.mean(Ytrain), np.var(Ytrain))
     
    return mse, nmse, mll, nmll, yfit, var

def eval_pooled_transfer_reg(train, test, background, algo):
    Xtrain = train[0]
    Ytrain = train[1]

    Xtest = test[0]
    Ytest = test[1]
    
    Xbgr = background[0]
    Ybgr = background[1]
    
   
    
    #Xtrain = np.vstack((Xtrain, Xbgr))
        
    Xtrain_merged = np.r_[Xtrain, Xbgr]
    Ytrain_merged = np.r_[Ytrain, Ybgr]
    
    algo.fit(Xtrain_merged, Ytrain_merged)
    yfit, var = algo.predict(Xtest, ret_var=True)
    print 'hyperparams={0}'.format(algo.hyperparams)
    
    mse = metric.mspe(Ytest, yfit)
    nmse = mse/np.var(Ytest)
    mll = metric.nlp(Ytest, yfit, var)
    nmll = mll-metric.nlpp(Ytest, np.mean(Ytrain), np.var(Ytrain))
     
    return mse, nmse, mll, nmll, yfit, var

def eval_mtl_gp_transfer(train, test, background, kernel):
    Xtrain = train[0]
    Ytrain = train[1]

    Xtest = test[0]
    Ytest = test[1]
    
    Xbgr = background[0]
    Ybgr = background[1]
    
    selector = RandomSubsetSelector(15) 


    #selector = KMeansSelector(15, False)
    #Xu = selector.apply(Xtrain, Ytrain)
    #selector = FixedSelector(Xu)
    
    itask = np.array([0, len(Xtrain)])
        
    Xtrain_merged = np.r_[Xtrain, Xbgr]
    Ytrain_merged = np.r_[Ytrain, Ybgr]
    
    selector = KMeansSelector(15, False)
    Xu = selector.apply(Xtrain_merged, Ytrain_merged)
    selector = FixedSelector(Xu)    
    
    
    #algo = SparseCMOGPRegression(kernel, beta=100, infer_method=SparseCMOGPExactInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=False)
    algo = SparseCMOGPRegression(kernel, infer_method=SparseCMOGPExactInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=True)
    algo.fit(Xtrain_merged, Ytrain_merged, itask)
    yfit, var = algo.predict_task(Xtest, 0, ret_var=True)
    print 'hyperparams={0}'.format(np.exp(algo.hyperparams))
    
    mse = metric.mspe(Ytest, yfit)
    nmse = mse/np.var(Ytest)
    mll = metric.nlp(Ytest, yfit, var)
    nmll = mll-metric.nlpp(Ytest, np.mean(Ytrain), np.var(Ytrain))
     
    return mse, nmse, mll, nmll, yfit, var


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
    fold_filename = '//home/marcel/datasets/multilevel/nga/ssa/transfer/transfer_nga_10eq_splitz/nga_{0}_indexes.mat'
    #fold_filename = '//home/marcel/datasets/multilevel/nga/ssa/transfer/mag_splitz/nga_{0}_indexes.mat'
    
    filename = '/home/marcel/datasets/multilevel/nga/ssa/transfer/transfer_eunga30.mat'
    #filename = '/home/marcel/datasets/multilevel/nga/ssa/transfer/transfer_ngaeu.mat'
    
    Xt,yt,Xb,yb = loadmat_transfer_data(filename)
    #for simple pitc approximation
    #eqs = X[:,0]
   
    
    n = Xt.shape[0]
    ypred = np.zeros((n,2)) #matrix of the fitted value and its variance
        
    mse = np.empty(nfolds)
    mll = np.empty(nfolds)
    nmse = np.empty(nfolds)
    nmll = np.empty(nfolds)
    weights = np.empty(nfolds)
    
    for i in xrange(nfolds):
        train, test = loadmat_folds(fold_filename.format(i+1))
        
        data_train = (Xt[train], yt[train])
        data_test = (Xt[test], yt[test])
        data_bgr = (Xb, yb)
           
        
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
        
        
        #algo = EMBayesRegression(alpha0=1, beta0=1, weight_bias=True)
        #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(0.001)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + PolynomialKernel(2, np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5)) + PolynomialKernel(2, np.log(1), np.log(1))
        #kernel = RBFKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #kernel = RBFKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(1)*np.ones(len(l)), np.log(1)) + NoiseKernel(np.log(0.5))
        kernel = ARDSEKernel(np.log(1)*np.ones(len(l)), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #kernel = ARDSEKernel(np.log(1)*np.ones(len(l)), np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(len(l)), np.log(1)) + NoiseKernel(np.log(0.5))
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
        algo = GPRegression(kernel, infer_method=ExactInference)
        #algo = PITCSparseGPRegression(igroup, kernel, noise_kernel, infer_method=PITCExactInference, selector=selector, fix_inducing=False)
        #algo = GPRegression(kernel, meanfct=meanfct, infer_method=ExactInference)
        weights[i] = len(test)
        
        #mse[i], nmse[i], mll[i], nmll[i], yfit, var = eval_reg(data_train, data_test, algo)
        #mse[i], nmse[i], mll[i], nmll[i], yfit, var = eval_stl_transfer_reg(data_train, data_test, data_bgr, algo)
        #mse[i], nmse[i], mll[i], nmll[i], yfit, var = eval_pooled_transfer_reg(data_train, data_test, data_bgr, algo)
        
        
        latent_kernel = ExpGaussianKernel(np.log(0.1))
        #latent_kernel = ExpARDGaussianKernel(np.ones(7)*np.log(0.1))
        #latent_kernel = CompoundKernel([ExpGaussianKernel(np.log(0.01)), ExpGaussianKernel(np.log(0.1))])
        #latent_kernel = DiracConvolvedKernel(FixedParameterKernel(SEKernel(np.log(0.1),np.log(1)), [1]))       
        #latent_kernel = DiracConvolvedKernel(FixedParameterKernel(SEKernel(np.log(0.01),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel(), [1]))
        #latent_kernel = DiracConvolvedKernel(FixedParameterKernel(ARDSEKernel(np.ones(7)*np.log(0.01),np.log(1)), [7]))
        #latent_kernel = CompoundKernel([DiracConvolvedKernel(FixedParameterKernel(ARDSEKernel(np.ones(7)*np.log(0.01),np.log(1)), [7])), DiracConvolvedKernel(FixedParameterKernel(ARDSEKernel(np.random.random(7)+0.0001,np.log(1)), [7]))])
        #latent_Kernel = DiracConvolvedKernel(GaussianKernel(np.log(1)))
        #noise_kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        #noise_kernel = ARDSEKernel(np.ones(7)*np.log(1),np.log(1))+ SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.3))
        noise_kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
        #noise_kernel = TaskNoiseKernel((periods-np.min(periods))/(np.max(periods)-np.min(periods)), 7, np.log(0.5))
        #noise_kernel = TaskNoiseKernel((periods-np.min(periods))/(np.max(periods)-np.min(periods)), 7, np.log(0.5))
        
        theta = [np.log(0.1), np.log(1)]
        #theta = np.r_[np.ones(7)*np.log(0.1), np.log(1)]
        #theta = [np.log(1)]
        #theta = [np.log(1), np.log(1)]
        #theta = [np.log(0.01), np.log(1)]
        
        kernel = ConvolvedMTLKernel(latent_kernel, theta, 2, noise_kernel) 

        
        mse[i], nmse[i], mll[i], nmll[i], yfit, var = eval_mtl_gp_transfer(data_train, data_test, data_bgr, kernel)
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

