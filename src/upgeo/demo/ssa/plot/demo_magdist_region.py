'''
Created on Mar 28, 2013

@author: marcel
'''
import numpy as np 
import upgeo.util.metric as metric

from upgeo.base.kernel import GroupNoiseKernel, HiddenKernel,\
    MaskedFeatureKernel, ARDSEKernel, NoiseKernel, DiracConvolvedKernel,\
    FixedParameterKernel, SEKernel, SqConstantKernel, LinearKernel,\
    ARDSELinKernel, ExpGaussianKernel, ExpARDGaussianKernel,\
    MaskedFeatureConvolvedKernel
from upgeo.util.filter import MeanShiftFilter, MinMaxFilter, FunctionFilter,\
    CompositeFilter
from upgeo.util.array import unique
from upgeo.demo.util import loadmat_mtl_data
from upgeo.base.selector import KMeansSelector, FixedSelector
from upgeo.mtl.kernel import ConvolvedMTLKernel
from upgeo.mtl.gp import SparseCMOGPRegression, STLGPRegression,\
    PooledGPRegression
from upgeo.mtl.infer import SparseCMOGPExactInference
from upgeo.util.glob import APPROX_TYPE
from upgeo.base.infer import ExactInference

def create_mtlgp_model(train, test, task_ids):
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


def create_testset(mag_idx, dist_idx, values):
    mag, dist = np.mgrid[4:8.1:0.1, 0:201]
    mag = mag.flatten()
    dist = dist.flatten()
    n = len(mag)
    
    X = np.tile(values, (n,1))
    if mag_idx < dist_idx:
        #np.vstack((X[0:mag_idx], mag, X[mag_idx:]))
        X = np.c_[X[:,0:mag_idx], mag, X[:,mag_idx:]]
        X = np.c_[X[:,:dist_idx], dist, X[:,dist_idx:]]
    else:
        X = np.c_[X[:,:dist_idx], dist, X[:,dist_idx:]]
        X = np.c_[X[:,0:mag_idx], mag, X[:,mag_idx:]]
     
    return X

if __name__ == '__main__':
    filename = '/home/mhermkes/datasets/multilevel/nga/ssa/transfer/viz_mtl_eudata_big.mat'
    #filename = '/home/mhermkes/datasets/multilevel/nga/ssa/transfer/viz_mtl_eudata_big_eq.mat'
    
    mag_idx = 0
    dist_idx = 5
    
    X,y,tasks = loadmat_mtl_data(filename)
    task_ids, itask = unique(tasks, True)
    k = len(task_ids)
    Xt = create_testset(mag_idx, dist_idx, [1,0,0,10,760])
    
    print X
    
    jbd_trans_fun = lambda x: np.log(np.sqrt(x**2 + 12**2))
    jbd_inv_fun = lambda x: np.sqrt(np.exp(x)**2 - 12**2)
    
    
    #event_idx = 0   #index of the event id row
    #site_idx = 1    #index of the site id row
    
    #event_mask = [0,1]    #mask of the event features, which should be normalized
    #site_mask = [6]            #mask of the site features, which should be normalized
    #record_mask = [5]    #mask of the record features, which should be normalized
    norm_mask = [0,4,5,6] 
    dist_mask =  [5] 
    #norm_mask = [1,5,6,7]
    #dist_mask = [6]
    
    fmask = np.r_[0, np.ones(7)]
    fmask = np.array(fmask, dtype=np.bool)
    
    
    dist_filter = FunctionFilter(jbd_trans_fun, jbd_inv_fun, dist_mask)
    cov_filter = MinMaxFilter(norm_mask)
    cov_filter = CompositeFilter([dist_filter, MinMaxFilter(norm_mask)])
    target_filter = MeanShiftFilter()
    
    
    #norm
    Xtrain = cov_filter.process(X)
    ytrain = np.squeeze(target_filter.process(y[:,np.newaxis]))
    Xtest = cov_filter.process(Xt, reuse_stats=True)
    
    
    #learn GP
    #l = (np.max(X,0)-np.min(X,0))/2
    #l[l == 0] = 1e-4   
    
    
    #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel()# + NoiseKernel(np.log(0.5))
    #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(0.001)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    #kernel = SEKernel(np.log(1), np.log(1))# + NoiseKernel(np.log(0.5))
    #kernel = RBFKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = RBFKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(1)*np.ones(7), np.log(1)) #+ NoiseKernel(np.log(0.5))
    kernel = ARDSEKernel(np.log(1)*np.ones(7), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() #+ NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(l), np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(len(l)), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDSELinKernel(np.log(l), np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDRBFKernel(np.log(l), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDRBFKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    #selector = KMeansSelector(30, False) 
    
    #kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask) + CorrelatedNoiseKernel(0, np.log(0.1), np.log(0.5))
    #kernel = MaskedFeatureKernel(ARDSEKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask) + CorrelatedNoiseKernel(0, np.log(0.1), np.log(0.5))
    
    #meanfunctions for standard data
    #meanfct = create_meanfct(7, data=None, mask=None) #mean
    #meanfct = create_meanfct(7, data=(Xtrain,ytrain), mask=None) #fixmean
    
    #meanfunctions for different parameters in the meanfct and covfct
    #meanfct = create_meanfct(10, data=None, mask=None) #mean
    #meanfct = create_meanfct(10, data=data_train, mask=None) #fixmean
    #kernel = MaskedFeatureKernel(kernel, fmask)
    
    #create complex noise model
    #noise_kernel = create_noise_kernel(0, np.log(1)) + NoiseKernel(np.log(0.5))
    noise_kernel = NoiseKernel(np.log(0.5))
    kernel = kernel + noise_kernel
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
    
    #algo = GPRegression(kernel, meanfct=meanfct, infer_method=ExactInference)

    #create kernel
    #kernel = SEKernel(np.log(np.mean(ll)), np.log(1)) + NoiseKernel(np.log(0.1))
    
    
   
        
    gp = STLGPRegression(kernel, infer_method=ExactInference)
    #gp = PooledGPRegression(kernel, infer_method=ExactInference)
    
    #selector = RandomSubsetSelector(15)
    selector = KMeansSelector(30, False) 
    Xu = selector.apply(Xtrain, ytrain)
    selector = FixedSelector(Xu)
    #
    
    
    #latent_kernel = ExpGaussianKernel(np.log(0.1))
    latent_kernel = ExpARDGaussianKernel(np.ones(7)*np.log(0.1))
    #latent_kernel = CompoundKernel([ExpGaussianKernel(np.log(0.1)), ExpGaussianKernel(np.log(0.2))])
    #latent_kernel = DiracConvolvedKernel(FixedParameterKernel(SEKernel(np.log(0.1),np.log(1)), [1]))       
    #latent_kernel = DiracConvolvedKernel(FixedParameterKernel(SEKernel(np.log(0.01),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel(), [1]))
    #latent_kernel = DiracConvolvedKernel(FixedParameterKernel(ARDSEKernel(np.ones(7)*np.log(0.1),np.log(1))+ SqConstantKernel(np.log(1)) * LinearKernel(), [7]))
    #latent_kernel = CompoundKernel([DiracConvolvedKernel(FixedParameterKernel(ARDSEKernel(np.ones(7)*np.log(0.1),np.log(1)), [7])), DiracConvolvedKernel(FixedParameterKernel(ARDSEKernel(np.ones(7)*np.log(0.25),np.log(1)), [7]))])
    #latent_kernel = CompoundKernel([ExpARDGaussianKernel(np.ones(7)*np.log(0.1)), ExpARDGaussianKernel(np.log(np.random.random(7)+0.0001))])
    #latent_kernel = CompoundKernel([ExpARDGaussianKernel(np.ones(7)*np.log(0.1)), ExpARDGaussianKernel(np.ones(7)*np.log(0.2))])
    #latent_Kernel = DiracConvolvedKernel(GaussianKernel(np.log(1)))
    #noise_kernel = SEKernel(np.log(0.1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() #+ NoiseKernel(np.log(0.5))
    noise_kernel = ARDSEKernel(np.ones(7)*np.log(0.1),np.log(1))+ SqConstantKernel(np.log(1)) * LinearKernel()# + NoiseKernel(np.log(0.5))
    #noise_kernel = ARDSEKernel(np.ones(7)*np.log(0.1),np.log(1))#+ NoiseKernel(np.log(0.5))
    #noise_kernel = SEKernel(np.log(0.1), np.log(1)) + NoiseKernel(np.log(0.5))
    #noise_kernel = TaskNoiseKernel((periods-np.min(periods))/(np.max(periods)-np.min(periods)), 7, np.log(0.5))
    #noise_kernel = TaskNoiseKernel((periods-np.min(periods))/(np.max(periods)-np.min(periods)), 7, np.log(0.5))
    
    noise_kernel = noise_kernel + NoiseKernel(np.log(0.5))
    #noise_kernel = MaskedFeatureKernel(noise_kernel, fmask) + create_noise_kernel(0, np.log(1)) + NoiseKernel(np.log(0.5))
    #latent_kernel = MaskedFeatureConvolvedKernel(latent_kernel, fmask)
    #theta = [np.log(0.1), np.log(1)]
    #theta = [np.log(0.1), np.log(1), np.log(0.2), np.log(1)]
    theta = np.r_[np.ones(7)*np.log(0.1), np.log(1)]
    #theta = np.r_[np.ones(7)*np.log(0.1), np.log(1), np.ones(7)*np.log(0.2), np.log(1)]
    #theta = [np.log(1)]
    #theta = [np.log(1),np.log(1)]
    #theta = [np.log(1), np.log(1)]
    #theta = [np.log(0.01), np.log(1)]   
    #kernel = ConvolvedMTLKernel(latent_kernel, theta, k, noise_kernel) 
    #idx = [7,15]
    #kernel._theta[:,idx] = np.log(np.random.rand(k,len(idx)))
    
    #gp = SparseCMOGPRegression(kernel, beta=100, infer_method=SparseCMOGPExactInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=True)        
    #gp = SparseCMOGPRegression(kernel, infer_method=SparseCMOGPExactInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=True)
    
    print 'X={0}'.format(X)
    print 'Xtest={0}'.format(Xtest)
    
    
    gp.fit(Xtrain,ytrain,itask)
    k = len(task_ids)
    yhat = np.zeros(len(X))
    for i in xrange(k):
        yfit, var = gp.predict_task(Xtest, q=i, ret_var=True)
        #print 'yfit={0}'.format(yfit)
        yhat[tasks==task_ids[i]] = gp.predict_task(Xtrain[tasks==task_ids[i]], q=i, ret_var=False)
        yfit = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
            
        #np.savetxt('/home/mhermkes/datasets/multilevel/nga/ssa/transfer/viz/region_model/ardselin/stl_region{0}.csv'.format(task_ids[i]), np.c_[Xt[:,[mag_idx, dist_idx]], yfit,var], delimiter=',')
    
    resid = yhat - ytrain
    #np.savetxt('/home/mhermkes/datasets/multilevel/nga/ssa/transfer/viz/region_model/resid/ardselin/stl_resid.csv', np.c_[tasks, X[:,[mag_idx, dist_idx]], resid], delimiter=',')
    print 'likel: {0}'.format(gp.log_likel)
    print 'train error: {0}'.format(metric.mspe(ytrain, yhat))
    print 'hyper params: {0}'.format(np.exp(gp.hyperparams))
    
