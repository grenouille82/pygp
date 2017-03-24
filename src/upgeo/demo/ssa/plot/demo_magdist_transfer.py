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
from upgeo.demo.util import loadmat_mtl_data, loadmat_transfer_data,\
    loadmat_folds
from upgeo.base.selector import KMeansSelector, FixedSelector
from upgeo.mtl.kernel import ConvolvedMTLKernel
from upgeo.mtl.gp import SparseCMOGPRegression, STLGPRegression,\
    PooledGPRegression
from upgeo.mtl.infer import SparseCMOGPExactInference
from upgeo.util.glob import APPROX_TYPE
from upgeo.base.infer import ExactInference
from upgeo.base.gp import GPRegression

def eval_stl_transfer_reg(train, test, background, Xviz, algo):
    Xtrain = train[0]
    ytrain = train[1]

    Xtest = test[0]
    ytest = test[1]
 
    algo.fit(Xtrain, ytrain)
    
    print 'likel: {0}'.format(gp.log_likel)
    yfit, var = gp.predict(Xtest, ret_var=True)
    resid_test = yfit-ytest
    print 'test error: {0}'.format(metric.mspe(ytest, yfit))
    yfit, var = gp.predict(Xtrain, ret_var=True)
    resid_train = yfit-ytrain
    print 'train error: {0}'.format(metric.mspe(ytrain, yfit))
    print 'hyper params: {0}'.format(np.exp(gp.hyperparams))
    yfit, var = gp.predict(Xviz, ret_var=True)
       
    return yfit, var, resid_train, resid_test

def eval_pooled_transfer_reg(train, test, background, Xviz, algo):
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
    print 'likel: {0}'.format(gp.log_likel)
    yfit, var = gp.predict(Xtest, ret_var=True)
    resid_test = yfit-ytest
    print 'test error: {0}'.format(metric.mspe(ytest, yfit))
    yfit, var = gp.predict(Xtrain, ret_var=True)
    resid_train = yfit-ytrain
    print 'train error: {0}'.format(metric.mspe(ytrain, yfit))
    print 'hyper params: {0}'.format(np.exp(gp.hyperparams))     
    yfit, var = gp.predict(Xviz, ret_var=True)   
    return yfit, var, resid_train, resid_test

def eval_mtl_gp_transfer(train, test, background, Xviz, kernel):
    Xtrain = train[0]
    Ytrain = train[1]

    Xtest = test[0]
    Ytest = test[1]
    
    Xbgr = background[0]
    Ybgr = background[1]
    
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
    #Xtrain = np.vstack((Xtrain, Xbgr))
    Xtrain_merged = np.r_[Xtrain, Xbgr]
    Ytrain_merged = np.r_[Ytrain, Ybgr]
    algo.fit(Xtrain_merged, Ytrain_merged)
    print 'likel: {0}'.format(gp.log_likel)
    yfit, var = gp.predict_task(Xtest, q=0, ret_var=True)
    resid_test = yfit-ytest
    print 'test error: {0}'.format(metric.mspe(ytest, yfit))
    yfit, var = gp.predict_task(Xtrain, q=0, ret_var=True)
    resid_train = yfit-ytrain
    print 'train error: {0}'.format(metric.mspe(ytrain, yfit))
    print 'hyper params: {0}'.format(np.exp(gp.hyperparams))  
    yfit, var = gp.predict_task(Xviz, q=0, ret_var=True)   
    return yfit, var, resid_train, resid_test

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
    eq_number = 30
    fold_number = 2
    fold_filename = '/home/marcel/datasets/multilevel/nga/ssa/transfer/transfer_nga_{0}eq_splitz/nga_{1}_indexes.mat'.format(eq_number, fold_number)
    print fold_filename
    #fold_filename = '//home/mhermkes/datasets/multilevel/nga/ssa/transfer/mag_splitz/nga_{0}_indexes.mat'
    
    #filename = '/home/mhermkes/datasets/multilevel/nga/ssa/transfer/transfer_ngasim.mat'
    #filename = '/home/mhermkes/datasets/multilevel/nga/ssa/transfer/transfer_ngaeu.mat'
    filename = '/home/marcel/datasets/multilevel/nga/ssa/transfer/viz_transfer_eunga30.mat'
    Xt,yt,Xb,yb = loadmat_transfer_data(filename)
    
    nt = len(Xt)
    X = np.r_[Xt,Xb]
    y = np.r_[yt,yb]
    
    mag_idx = 0
    dist_idx = 5
    
    Xviz = create_testset(mag_idx, dist_idx, [1,0,0,10,760])
    
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
    Xnorm = cov_filter.process(X)
    ynorm = np.squeeze(target_filter.process(y[:,np.newaxis]))
    Xviznorm = cov_filter.process(Xviz, reuse_stats=True)
    
    train, test = loadmat_folds(fold_filename)
    
    Xtrain = Xnorm[train]
    ytrain = ynorm[train]
    Xtest = Xnorm[test]
    ytest = ynorm[test]
    Xbgr = Xnorm[nt:]
    ybgr = ynorm[nt:]
    
    Xmerged = np.r_[Xtrain, Xbgr]
    ymerged = np.r_[ytrain, ybgr]
    itask = np.array([0, len(Xtrain)])
    
    data_train = (Xtrain, ytrain)
    data_test = (Xtest, ytest)
    data_bgr = (Xbgr, ybgr)
    
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
    
    #kernel = MaskedFeatureKernel(kernel, fmask)
    
    #create complex noise model
    #noise_kernel = create_noise_kernel(0, np.log(1)) + NoiseKernel(np.log(0.5))
    noise_kernel = NoiseKernel(np.log(0.5))
    kernel = kernel + noise_kernel
    #noise_kernel = create_noise_kernel(0, np.log(1), MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[np.zeros(6), np.ones(2)], dtype=np.bool)))
    #noise_kernel = create_noise_kernel(0, np.log(1), MaskedFeatureKernel(ARDSEKernel(np.log(l[6:7]), np.log(1)), np.array(np.r_[np.zeros(6), np.ones(2)], dtype=np.bool)))
    #kernel = MaskedFeatureKernel(kernel, fmask) + noise_kernel
    
    #create kernel
    #kernel = SEKernel(np.log(np.mean(ll)), np.log(1)) + NoiseKernel(np.log(0.1))
    
    gp = GPRegression(kernel, infer_method=ExactInference)
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
    kernel = ConvolvedMTLKernel(latent_kernel, theta, 2, noise_kernel) 
    #idx = [7,15]
    #kernel._theta[:,idx] = np.log(np.random.rand(k,len(idx)))
    
    #gp = SparseCMOGPRegression(kernel, beta=100, infer_method=SparseCMOGPExactInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=True)        
    #gp = SparseCMOGPRegression(kernel, infer_method=SparseCMOGPExactInference, approx_type=APPROX_TYPE.PITC, selector=selector, fix_inducing=True)
    
    #yfit, var, resid_train, resid_test =  eval_stl_transfer_reg(data_train, data_test, data_bgr, Xviznorm, gp)
    #yfit, var, resid_train, resid_test = eval_pooled_transfer_reg(data_train, data_test, data_bgr, Xviznorm, gp)
    yfit, var, resid_train, resid_test = eval_mtl_gp_transfer(data_train, data_test, data_bgr,  Xviznorm, kernel)
    
    #print 'X={0}'.format(X)
    #print 'Xtest={0}'.format(Xtest)
    
    
    #yfit, var = gp.predict_task(Xviznorm, q=0, ret_var=True)
    yfit = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
    np.savetxt('/home/mhermkes/datasets/multilevel/nga/ssa/transfer/viz/transfer/ardselin/{0}/stl_gp_predict.csv'.format(eq_number), np.c_[Xviz[:,[mag_idx, dist_idx]], yfit,var], delimiter=',')
    np.savetxt('/home/mhermkes/datasets/multilevel/nga/ssa/transfer/viz/transfer/resid/ardselin/{0}/stl_gp_resid_test.csv'.format(eq_number), np.c_[Xt[test,[mag_idx, dist_idx]], resid_test], delimiter=',')
    np.savetxt('/home/mhermkes/datasets/multilevel/nga/ssa/transfer/viz/transfer/resid/ardselin/{0}/stl_gp_resid_train.csv'.format(eq_number), np.c_[Xt[test,[mag_idx, dist_idx]], resid_train], delimiter=',')
    
