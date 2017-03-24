'''
Created on Dec 1, 2012

@author: marcel
'''
import numpy as np

from upgeo.demo.eupaper.util import load_train_data
from upgeo.util.filter import BagFilter, StandardizeFilter, FunctionFilter,\
    CompositeFilter, MeanShiftFilter
from upgeo.base.kernel import SEKernel, SqConstantKernel, LinearKernel,\
    GroupNoiseKernel, NoiseKernel, MaskedFeatureKernel
from upgeo.base.infer import ExactInference, OnePassInference
from upgeo.base.gp import GPRegression

if __name__ == '__main__':
    #train_fname = '/home/marcel/datasets/multilevel/eusinan/bssa/comparison_paper/eudata_train.csv'
    train_fname = '/home/marcel/datasets/multilevel/eusinan/bssa/comparison_paper/eudata_train_fullperiods.csv'
    
    pred_cols = np.arange(0,9)
    #target_cols = np.arange(9,19)
    target_cols = np.arange(9,73)#for full periods

    vs30 = np.arange(100,2010,10)
    distances = [10,100]
    magnitudes = [6,7] 
    dist_idx = 5
    vs30_idx = 6
    n = len(vs30)
    #covariates = [7,15,0,1,0] #mw=7,fdepth=15,fm=ss
    #covariates = [6,15,0,1,0] #mw=7,fdepth=15,fm=ss
    covariates = [15,0,1,0] #mw=7,fdepth=15,fm=ss


    
    Xtrain,Ytrain = load_train_data(train_fname, pred_cols, target_cols)
    ytrain = Ytrain[:,18] #pga
    #ytrain = Ytrain[:,6] #T1

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
    
    
    Xtrain = cov_filter.process(Xtrain)
    ytrain = np.squeeze(target_filter.process(ytrain[:,np.newaxis]))

    print event_bag.shape
    print Xtrain.shape

    #for corr noise model    
    Xtrain = np.c_[event_bag, Xtrain]


    kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() 
    noise_kernel = GroupNoiseKernel(0, np.log(0.5)) + NoiseKernel(np.log(0.5))
    kernel = MaskedFeatureKernel(kernel, fmask) + noise_kernel
    
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
            
            np.savetxt('/home/marcel/datasets/multilevel/eusinan/bssa/comparison_paper/fig5/gpselincorr_mag{0}_rjb{1}_pred_T02.csv'.format(mw,r), np.c_[vs30,yfit], delimiter=',')        
        
        
        
    
    
    