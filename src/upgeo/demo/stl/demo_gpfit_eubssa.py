'''
Created on Nov 19, 2012

@author: marcel
'''

import upgeo.util.metric as metric
import numpy as np
import scipy.io as sio
from upgeo.util.filter import BagFilter, StandardizeFilter, FunctionFilter,\
    CompositeFilter, MeanShiftFilter
from upgeo.base.gp import GPRegression
from upgeo.base.infer import ExactInference, OnePassInference
from upgeo.base.kernel import FixedParameterKernel, SEKernel, SqConstantKernel,\
    LinearKernel, NoiseKernel, DiracConvolvedKernel, ARDSEKernel,\
    MaskedFeatureKernel, GroupNoiseKernel, HiddenKernel
from upgeo.mtl.kernel import ConvolvedMTLKernel
from upgeo.base.mean import MaskedFeatureMean, BiasedLinearMean, HiddenMean
from upgeo.regression.linear import LSRegresion

def load_train_data(fname, pred_cols, target_cols):
    data = np.loadtxt(fname, delimiter=',', skiprows=1)
    X = data[:, pred_cols]
    Y = data[:, target_cols]
    
    return X,Y
    
def load_test_data(fname):
    mat_dict = sio.loadmat(fname)
    X = mat_dict['X']
    return X


def create_meanfct(nfeatures, data=None, mask=None):
    meanfct = None
    
    if data != None:
        rmodel = LSRegresion()
        rmodel.fit(data[0], data[1])
        meanfct = BiasedLinearMean(rmodel.weights, rmodel.intercept)
        meanfct = HiddenMean(meanfct)
    else:
        meanfct = BiasedLinearMean(np.zeros(nfeatures), 0)
    
    if mask != None:
        meanfct = MaskedFeatureMean(meanfct, mask)

    return meanfct

def create_noise_kernel(grp_idx, s, kernel=None, mask=None):
    noise_kernel = GroupNoiseKernel(grp_idx, s)
    if kernel != None:
        noise_kernel = HiddenKernel(noise_kernel)
        noise_kernel = noise_kernel*kernel
    if mask != None:
        noise_kernel = MaskedFeatureKernel(noise_kernel, mask)
    return noise_kernel


def create_mtl_kernel(ntasks, nfeatures, l=None, fmask=None, grp_idx=None):
    #construct mtl kernel
    
    if fmask == None:
        
        #lKernel = FixedParameterKernel(SEKernel(np.log(1),np.log(1)), [1])
        lKernel = FixedParameterKernel(SEKernel(np.log(1),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel(), [1])
        #lKernel = SEKernel(np.log(1),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel()
        #lKernel = FixedParameterKernel(ARDSEKernel(np.log(l),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel(), [len(l)])
        #lKernel = FixedParameterKernel(ARDSEKernel(np.log(l),np.log(1)), [len(l)])
        dpKernel = DiracConvolvedKernel(lKernel)
        #idpKernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
        idpKernel = ARDSEKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    else:
        
        lKernel = FixedParameterKernel(MaskedFeatureKernel(SEKernel(np.log(1),np.log(1)), fmask), [1])
        #lKernel = FixedParameterKernel(MaskedFeatureKernel(SEKernel(np.log(1),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel(), fmask), [1])
        #lKernel = SEKernel(np.log(1),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel()
        #lKernel = FixedParameterKernel(MaskedFeatureKernel(ARDSEKernel(np.log(l),np.log(1))+SqConstantKernel(np.log(1)) * LinearKernel(), fmask), [len(l)])
        #lKernel = FixedParameterKernel(MaskedFeatureKernel(ARDSEKernel(np.log(l),np.log(1)), fmask), [len(l)])
    
        dpKernel = DiracConvolvedKernel(lKernel)
        #idpKernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask)
        idpKernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask) + NoiseKernel(np.log(0.5))
        #idpKernel = MaskedFeatureKernel(ARDSEKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask)+ NoiseKernel(np.log(0.5))
    
    if grp_idx != None:
        idpKernel = idpKernel + GroupNoiseKernel(grp_idx, np.log(0.1))
        #idpKernel = idpKernel + CorrelatedNoiseKernel(grp_idx, np.log(0.1), np.log(0.5))
    
    #theta = np.r_[np.log(0.1)*np.ones(nfeatures), np.log(1)]
    theta = [np.log(0.1)]
    kernel = ConvolvedMTLKernel(dpKernel, theta, ntasks, idpKernel)
    return kernel


if __name__ == '__main__':
    
    #load and preprocess data 
    #data_fname = '/home/marcel/datasets/multilevel/nga/pooled/mainshock/viz_nga_mainshock1.csv'
    #data_fname = '/home/marcel/datasets/multilevel/nga/pooled/mainshock/viz_nga_mainshock5.csv'
    #data_fname = '/home/marcel/datasets/multilevel/nga/pooled/50/viz_nga_pga50_1.csv'
    train_fname = '/home/mhermkes/datasets/multilevel/eusinan/bssa/viz/viz_eudata_train.csv'
    test_fname = '/home/mhermkes/datasets/multilevel/eusinan/bssa/viz/viz_eudata_test.mat'
    
    target_name = 'pgv'
    
    pred_cols = np.arange(0,9)
    target_cols = np.arange(9,19)
    
    Xtrain,Ytrain = load_train_data(train_fname, pred_cols, target_cols)
    Xtest = load_test_data(test_fname)
    ytrain = Ytrain[:,9]
    
    
    jbd_trans_fun = lambda x: np.log(np.sqrt(x**2 + 12**2))
    jbd_inv_fun = lambda x: np.sqrt(np.exp(x)**2 - 12**2)
    
    
    event_idx = 0   #index of the event id row
    site_idx = 1    #index of the site id row
    
    event_mask = [0,1]    #mask of the event features, which should be normalized
    site_mask = [6]            #mask of the site features, which should be normalized
    record_mask = [5]    #mask of the record features, which should be normalized
    dist_mask =  [5]
    
    data_mask = np.ones(Xtrain.shape[1], 'bool')
    data_mask[event_idx] = data_mask[site_idx] = 0
    
    #periodic_mask = []  #mask of periodic features
    
    fmask = np.r_[0, np.ones(7)]
    fmask = np.array(fmask, dtype=np.bool)
    
    event_bag = Xtrain[:,event_idx]
    site_bag = Xtrain[:,site_idx]
    Xtrain = Xtrain[:,data_mask]
    
    event_filter = BagFilter(event_bag, StandardizeFilter(1,event_mask))
    site_filter = BagFilter(site_bag, StandardizeFilter(1,site_mask))
    record_filter = StandardizeFilter(1, record_mask)
    #dist_filter = FunctionFilter(np.log, np.exp, dist_mask)
    dist_filter = FunctionFilter(jbd_trans_fun, jbd_inv_fun, dist_mask)
    #periodic_filter = FunctionFilter(np.cos, periodic_mask)
    
    cov_filter = CompositeFilter([dist_filter, event_filter, site_filter, record_filter])
    #cov_filter = CompositeFilter([event_filter, site_filter, record_filter])
    target_filter = MeanShiftFilter()
    
    
    #norm
    Xtrain = cov_filter.process(Xtrain)
    ytrain = np.squeeze(target_filter.process(ytrain[:,np.newaxis]))
    Xtest = cov_filter.process(Xtest, reuse_stats=True)
    #Xtrain = dist_filter.process(Xtrain)
    #Xtest = dist_filter.process(Xtest)

    #print Xtrain[0:10,:]

    #for corr noise model
    #print event_bag.shape
    Xtrain = np.c_[event_bag, Xtrain]
    Xtest = np.c_[np.ones(len(Xtest))*-1, Xtest]
    

    #learn GP
    l = (np.max(Xtrain,0)-np.min(Xtrain,0))/2
    l[l == 0] = 1e-4   
    
    
    #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(0.001)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = RBFKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = RBFKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    kernel = ARDSEKernel(np.log(1)*np.ones(len(l)-1), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(1)*np.ones(len(l)-1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(l), np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(len(l)), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDSELinKernel(np.log(l), np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDRBFKernel(np.log(l), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDRBFKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    #selector = KMeansSelector(30, False) 
    
    #kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask) + CorrelatedNoiseKernel(0, np.log(0.1), np.log(0.5))
    #kernel = MaskedFeatureKernel(ARDSEKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask) + CorrelatedNoiseKernel(0, np.log(0.1), np.log(0.5))
    
    #meanfunctions for standard data
    #meanfct = create_meanfct(7, data=None, mask=None) #mean
    meanfct = create_meanfct(7, data=(Xtrain,ytrain), mask=None) #fixmean
    
    #meanfunctions for different parameters in the meanfct and covfct
    #meanfct = create_meanfct(10, data=None, mask=None) #mean
    #meanfct = create_meanfct(10, data=data_train, mask=None) #fixmean
    #kernel = MaskedFeatureKernel(kernel, fmask)
    
    #create complex noise model
    noise_kernel = create_noise_kernel(0, np.log(1))
    #noise_kernel = create_noise_kernel(0, np.log(1), MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), np.array(np.r_[np.zeros(6), np.ones(2)], dtype=np.bool)))
    #noise_kernel = create_noise_kernel(0, np.log(1), MaskedFeatureKernel(ARDSEKernel(np.log(l[6:7]), np.log(1)), np.array(np.r_[np.zeros(6), np.ones(2)], dtype=np.bool)))
    kernel = MaskedFeatureKernel(kernel, fmask) + noise_kernel
    
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
    
    
    gp = GPRegression(kernel, infer_method=ExactInference)
    #gp = GPRegression(kernel, meanfct=meanfct, infer_method=ExactInference)
    #gp = GPRegression(kernel, infer_method=OnePassInference)
    gp.fit(Xtrain, ytrain)    
    
    
    n = len(Xtest)
    step = 100000
    Ypred = np.empty((len(Xtest),2))
    Ytrainpred = np.empty((len(Xtrain),2))
    
    Ytrainpred[:,0], Ytrainpred[:,1] = gp.predict(Xtrain, ret_var=True)
    yfit = Ytrainpred[:,0]
    
    print 'likel: {0}'.format(gp.log_likel)
    print 'train error: {0}'.format(metric.mspe(ytrain, yfit)/np.var(ytrain))
    print 'hyper params: {0}'.format(gp.hyperparams)
    Ytrainpred[:,0] = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
    np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/viz/train_fit/T4/gpardselin_corrnoise_pred.csv', Ytrainpred, delimiter=',')
#    
#    for i in np.arange(0,n,step):
#        if i+step > n:
#            Ypred[i:n,0],Ypred[i:n,1] = gp.predict(Xtest[i:n,:], ret_var=True)
#        else:
#            print len(Xtest[i:i+step,:])
#            Ypred[i:(i+step),0],Ypred[i:(i+step),1] = gp.predict(Xtest[i:i+step,:], ret_var=True)
#    
#    yfit = Ypred[:,0]
#    Ypred[:,0] = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
#    np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/viz/T4/gpse_corrnoise_pred.csv', Ypred, delimiter=',')
    
#    print 'likel: {0}'.format(gp.log_likel)
#    print 'train error: {0}'.format(metric.mspe(ytrain, gp.predict(Xtrain))/np.var(ytrain))
#    print 'hyper params: {0}'.format(gp.hyperparams)
    
    