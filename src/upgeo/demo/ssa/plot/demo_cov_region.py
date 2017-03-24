'''
Created on Apr 3, 2013

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
from upgeo.base.selector import KMeansSelector, FixedSelector,\
    RandomSubsetSelector
from upgeo.mtl.kernel import ConvolvedMTLKernel
from upgeo.mtl.gp import SparseCMOGPRegression, STLGPRegression,\
    PooledGPRegression
from upgeo.mtl.infer import SparseCMOGPExactInference,\
    SparseCMOGPOnePassInference
from upgeo.util.glob import APPROX_TYPE
from upgeo.base.infer import ExactInference
from upgeo.util.stats import covcorr


def create_noise_kernel(grp_idx, s, kernel=None, mask=None):
    noise_kernel = GroupNoiseKernel(grp_idx, s)
    if kernel != None:
        noise_kernel = HiddenKernel(noise_kernel)
        noise_kernel = noise_kernel*kernel
    if mask != None:
        noise_kernel = MaskedFeatureKernel(noise_kernel, mask)
    return noise_kernel


def create_testset(mags, mag_idx, values):
    
    #mag, dist = np.mgrid[4:8.1:0.5, [30, 110]]
    #mag = mag.flatten()
    #dist = dist.flatten()
    n = len(mags)
    X = np.tile(values, (n,1))
    X = np.c_[X[:,0:mag_idx], mags, X[:,mag_idx:]]
    X = np.array(X, dtype=np.float)
    return X

if __name__ == '__main__':
    filename = '/home/marcel/datasets/multilevel/nga/ssa/transfer/viz_mtl_eudata_big.mat'
    #filename = '/home/marcel/datasets/multilevel/nga/ssa/transfer/viz_mtl_eudata_big_eq.mat'

    mag_idx = 0
    dist_idx = 5
    
    X,y,tasks = loadmat_mtl_data(filename)
    task_ids, itask = unique(tasks, True)
    k = len(task_ids)
    
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
    #cov_filter = MinMaxFilter(norm_mask)
    cov_filter = CompositeFilter([dist_filter, MinMaxFilter(norm_mask)])
    target_filter = MeanShiftFilter()
    
    
    #norm
    Xtrain = cov_filter.process(X)
    ytrain = np.squeeze(target_filter.process(y[:,np.newaxis]))
    
    
    #learn GP
    #l = (np.max(X,0)-np.min(X,0))/2
    #l[l == 0] = 1e-4   
        
    selector = RandomSubsetSelector(15)
    #selector = KMeansSelector(15, False) 
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
    #noise_kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() #+ NoiseKernel(np.log(0.5))
    #noise_kernel = ARDSEKernel(np.ones(7)*np.log(0.1),np.log(1))+ SqConstantKernel(np.log(1)) * LinearKernel()# + NoiseKernel(np.log(0.5))
    #noise_kernel = ARDSEKernel(np.ones(7)*np.log(1),np.log(1))#+ NoiseKernel(np.log(0.5))
    noise_kernel = SEKernel(np.log(0.001), np.log(1)) #+ NoiseKernel(np.log(0.5))
    #noise_kernel = TaskNoiseKernel((periods-np.min(periods))/(np.max(periods)-np.min(periods)), 7, np.log(0.5))
    #noise_kernel = TaskNoiseKernel((periods-np.min(periods))/(np.max(periods)-np.min(periods)), 7, np.log(0.5))
    
    
    #noise_kernel = MaskedFeatureKernel(noise_kernel, fmask) + create_noise_kernel(0, np.log(1)) + NoiseKernel(np.log(0.5))
    noise_kernel = noise_kernel + NoiseKernel(np.log(np.sqrt(0.135335283236613)))
    #noise_kernel = NoiseKernel(np.log(0.5))
    #latent_kernel = MaskedFeatureConvolvedKernel(latent_kernel, fmask)
    #theta = [np.log(0.1), np.log(1)]
    #theta = [np.log(0.1), np.log(1), np.log(0.2), np.log(1)]
    theta = np.r_[np.ones(7)*np.log(0.1), np.log(1)]
    #theta = np.r_[np.ones(7)*np.log(0.1), np.log(1), np.ones(7)*np.log(0.2), np.log(1)]
    #theta = [np.log(1)]
    #theta = [np.log(1),np.log(1)]
    #theta = [np.log(1), np.log(1)]
    #theta = [np.log(0.01), np.log(1)]   
    kernel = ConvolvedMTLKernel(latent_kernel, theta, k, noise_kernel) 
    #idx = [7,15]
    #kernel._theta[:,idx] = np.log(np.random.rand(k,len(idx)))   
    
    gp = SparseCMOGPRegression(kernel, beta=100, infer_method=SparseCMOGPOnePassInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=False)        
    #gp = SparseCMOGPRegression(kernel, infer_method=SparseCMOGPExactInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=True)
    
    print 'X={0}'.format(X)
    print 'Xtrain={0}'.format(Xtrain)
    
    gp.fit(Xtrain,ytrain,itask)
    k = len(task_ids)
    yhat = np.zeros(len(X))
    for i in xrange(k):
        #print 'yfit={0}'.format(yfit)
        yhat[tasks==task_ids[i]] = gp.predict_task(Xtrain[tasks==task_ids[i]], q=i, ret_var=False)
        
    
    
    mags = np.array([4,5,6,7,8])
    Xt = create_testset(mags, mag_idx, [1,0,0,10,30,760])
    print 'Xt = {0}'.format(Xt)
    Xtest = cov_filter.process(Xt, reuse_stats=True)
    print 'Xtest = {0}'.format(Xtest)
    n = len(Xt)
    mu, Sigma = gp.posterior(Xtest)
    yfit, var = gp.predict(Xtest, True)
    print 'yfit={0}'.format(yfit)
    print 'mu={0}'.format(mu)
    print 'var={0}'.format(var)
    print 'Sigmadiag={0}'.format(np.diag(Sigma))
    print 'Sigma={0}'.format(Sigma)
    for i in xrange(len(mags)):
        region_cov = Sigma[i:n*k:n, i:n*k:n]
        region_corr = covcorr(region_cov)
        np.savetxt('/home/marcel/datasets/multilevel/nga/ssa/transfer/viz/region_model/corrmatrix/se/mag{0}_dist30_region_correlation.csv'.format(mags[i]), region_corr, delimiter=',')
    
    Xt = create_testset(mags, mag_idx, [1,0,0,10,100,760])
    Xtest = cov_filter.process(Xt, reuse_stats=True)
    
    _, Sigma = gp.posterior(Xtest)
    for i in xrange(len(mags)):
        region_cov = Sigma[i:n*k:n, i:n*k:n]
        region_corr = covcorr(region_cov)
        np.savetxt('/home/marcel/datasets/multilevel/nga/ssa/transfer/viz/region_model/corrmatrix/se/mag{0}_dist100_region_correlation.csv'.format(mags[i]), region_corr, delimiter=',')
    
    
    
    #np.savetxt('/home/marcel/datasets/multilevel/nga/ssa/transfer/viz/region_model/resid/ardselincorr/transfer_opt_resid.csv', np.c_[tasks, X[:,[mag_idx, dist_idx]], resid], delimiter=',')
    print 'likel: {0}'.format(gp.log_likel)
    print 'train error: {0}'.format(metric.mspe(ytrain, yhat))
    print 'hyper params: {0}'.format(np.exp(gp.hyperparams))
    

    