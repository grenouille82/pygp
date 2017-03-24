'''
Created on Dec 1, 2012

@author: marcel
'''
import numpy as np

from upgeo.demo.eupaper.util import load_train_data
from upgeo.util.filter import BagFilter, StandardizeFilter, FunctionFilter,\
    CompositeFilter, MeanShiftFilter
from upgeo.base.kernel import SEKernel, SqConstantKernel, LinearKernel,\
    GroupNoiseKernel, NoiseKernel, MaskedFeatureKernel, ARDSEKernel
from upgeo.base.infer import ExactInference, OnePassInference
from upgeo.base.gp import GPRegression

if __name__ == '__main__':
    #train_fname = '/home/marcel/datasets/multilevel/eusinan/bssa/comparison_paper/eudata_train.csv'
    train_fname = '/home/marcel/datasets/multilevel/nga/ssa/nga_ssa_train.csv'
    
    pred_cols = np.arange(0,9)
    #target_cols = np.arange(9,19)
    target_cols = np.arange(9,12)#for full periods

    vs30 = np.arange(100,1010,10)
    distances = [10,100]
    magnitudes = [6,7] 
    dist_idx = 5
    vs30_idx = 6
    n = len(vs30)
    #covariates = [7,15,0,1,0] #mw=7,fdepth=15,fm=ss
    #covariates = [6,15,0,1,0] #mw=7,fdepth=15,fm=ss
    covariates = [0,0,1,15] #mw=7,fdepth=15,fm=ss


    
    Xtrain,Ytrain = load_train_data(train_fname, pred_cols, target_cols)
    ytrain = Ytrain[:,0] #pga
    #ytrain = Ytrain[:,1] #T0.1
    #ytrain = Ytrain[:,2] #T0.5

    jbd_trans_fun = lambda x: np.log(np.sqrt(x**2 + 12**2))
    jbd_inv_fun = lambda x: np.sqrt(np.exp(x)**2 - 12**2)
    
    
    event_idx = 0   #index of the event id row
    site_idx = 1    #index of the site id row
    
    event_mask = [0,4]    #mask of the event features, which should be normalized
    site_mask = [6]            #mask of the site features, which should be normalized
    record_mask = [5]    #mask of the record features, which should be normalized
    dist_mask =  [5]
    
    data_mask = np.ones(Xtrain.shape[1], 'bool')
    data_mask[event_idx] = data_mask[site_idx] = 0
    
    #periodic_mask = []  #mask of periodic features
    
    
    fmask = np.r_[0, np.ones(7)]
    fmask = np.array(fmask, dtype=np.bool)
    
    
    print 'Xtrain'
    print Xtrain[0:2]
    print ytrain
    
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
    
    
    
    
    Xtrain = cov_filter.process(Xtrain)
    ytrain = np.squeeze(target_filter.process(ytrain[:,np.newaxis]))

    print event_bag.shape
    print Xtrain.shape

    #for corr noise model    
    Xtrain = np.c_[event_bag, Xtrain]


    #kernel = SEKernel(np.log(1), np.log(1)) 
    #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel()
    kernel = ARDSEKernel(np.log(1)*np.ones(7), np.log(1)) 
    noise_kernel = GroupNoiseKernel(0, np.log(0.5)) + NoiseKernel(np.log(0.5))
    kernel = MaskedFeatureKernel(kernel, fmask) + noise_kernel
    
    print 'Xtrain'
    print Xtrain[0:2]
    print ytrain
    
    gp = GPRegression(kernel, infer_method=ExactInference)
    gp.fit(Xtrain, ytrain)
    
    for r in distances:
        for mw in magnitudes:
            Xtest = np.tile(np.r_[mw,covariates,r], (n,1))
            
            Xtest = np.c_[Xtest, vs30]
            Xtest = np.array(Xtest, dtype=np.float)
            print Xtest
            Xtest = cov_filter.process(Xtest, reuse_stats=True)
            Xtest = np.c_[np.ones(len(Xtest))*-1, Xtest]
            yfit = gp.predict(Xtest)
            yfit = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
            
            np.savetxt('/home/marcel/datasets/multilevel/nga/ssa/viz/vs30scale/gpardsecorr_mag{0}_rjb{1}_pred_pga.csv'.format(mw,r), np.c_[vs30,yfit], delimiter=',')       
        
    
    print 'likel={0}'.format(gp.log_likel)
    print 'hyperparams={0}'.format(np.exp(gp.hyperparams))    
        
    
    
    