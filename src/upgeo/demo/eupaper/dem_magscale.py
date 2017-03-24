'''
Created on Dec 1, 2012

@author: marcel
'''
'''
Created on Dec 1, 2012

@author: marcel
'''
import numpy as np

from upgeo.demo.eupaper.util import load_train_data
from upgeo.util.filter import BagFilter, StandardizeFilter, FunctionFilter,\
    CompositeFilter, MeanShiftFilter
from upgeo.base.kernel import SEKernel, SqConstantKernel, LinearKernel,\
    GroupNoiseKernel, NoiseKernel, MaskedFeatureKernel, PolynomialKernel,\
    ARDSEKernel, ARDLinearKernel, ARDPolynomialKernel
from upgeo.base.infer import ExactInference, OnePassInference
from upgeo.base.gp import GPRegression

if __name__ == '__main__':
    train_fname = '/home/marcel/datasets/multilevel/eusinan/bssa/comparison_paper/eudata_train.csv'
    #train_fname = '/home/marcel/datasets/multilevel/eusinan/bssa/comparison_paper/eudata_train_mw2.csv'
    #train_fname = '/home/marcel/datasets/multilevel/eusinan/bssa/comparison_paper/eudata_train_fullperiods.csv'
    
    pred_cols = np.arange(0,9)
    target_cols = np.arange(9,19)
    
    #mag2 dataset
    #pred_cols = np.arange(0,10)
    #target_cols = np.arange(10,20)
    
    #target_cols = np.arange(9,73)#for full periods
    
    mag_idx = 0
    magnitudes = np.arange(4,8.1,0.1)
    n = len(magnitudes)
    covariates = [15,0,1,0,10,760] #fdepth=15,fm=ss,dist=30,vs30=760
    


    
    Xtrain,Ytrain = load_train_data(train_fname, pred_cols, target_cols)
    #ytrain = Ytrain[:,1] #PGA
    ytrain = Ytrain[:,10] #T3
    

    jbd_trans_fun = lambda x: np.log(np.sqrt(x**2 + 12**2))
    jbd_inv_fun = lambda x: np.sqrt(np.exp(x)**2 - 12**2)
    
    
    event_idx = 0   #index of the event id row
    site_idx = 1    #index of the site id row
    
    event_mask = [0,1]    #mask of the event features, which should be normalized
    site_mask = [6]            #mask of the site features, which should be normalized
    record_mask = [5]    #mask of the record features, which should be normalized
    dist_mask =  [5]
    
    #mask with mag2
    #event_mask = [0,1,2]    #mask of the event features, which should be normalized
    #site_mask = [7]            #mask of the site features, which should be normalized
    #record_mask = [6]    #mask of the record features, which should be normalized
    #dist_mask =  [6]
    
    
    data_mask = np.ones(Xtrain.shape[1], 'bool')
    data_mask[event_idx] = data_mask[site_idx] = 0
    
    #periodic_mask = []  #mask of periodic features
    
    
    fmask = np.r_[0, np.ones(7)]
    fmask = np.array(fmask, dtype=np.bool)
    
    #fmask_se = np.r_[0, 1, 0, np.ones(6)]
    #fmask_se = np.array(fmask_se, dtype=np.bool)
    #fmask_lin = np.r_[0, 0, np.ones(7)]
    #fmask_lin = np.array(fmask_lin, dtype=np.bool)
    
    
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


    #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel()
    #kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)), fmask_se) + MaskedFeatureKernel(SqConstantKernel(np.log(1)) * LinearKernel(), fmask_lin)
    #kernel = SEKernel(np.log(1), np.log(1))
    #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + PolynomialKernel(2, np.log(1), np.log(1))
    #kernel = SEKernel(np.log(1), np.log(1)) + PolynomialKernel(2, np.log(1), np.log(1))
    #kernel = ARDSEKernel(np.log(1)*np.ones(7), np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(7), np.log(1))
    #kernel = MaskedFeatureKernel(ARDSEKernel(np.log(1)*np.ones(7), np.log(1)), fmask_se) + MaskedFeatureKernel(ARDLinearKernel(np.log(1)*np.ones(7), np.log(1)), fmask_lin)
    #kernel = ARDSEKernel(np.log(1)*np.ones(7), np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(7), np.log(1)) + ARDPolynomialKernel(2, np.log(1)*np.ones(7), np.log(1), np.log(1))
    kernel = ARDSEKernel(np.log(1)*np.ones(7), np.log(1))+ SqConstantKernel(np.log(1)) * LinearKernel()
    #kernel = ARDSEKernel(np.log(1)*np.ones(7), np.log(1)) + ARDPolynomialKernel(2, np.log(1)*np.ones(7), np.log(1), np.log(1))
    #noise_kernel = GroupNoiseKernel(0, np.log(0.5)) + NoiseKernel(np.log(0.5))
    noise_kernel = NoiseKernel(np.log(0.5))
    #kernel = MaskedFeatureKernel(kernel, fmask) + noise_kernel
    kernel = kernel + noise_kernel
    
    print 'params before'
    print kernel.params
    
    gp = GPRegression(kernel, infer_method=ExactInference)
    gp.fit(Xtrain, ytrain)
    
    
    print 'params after'
    print kernel.params
    
    print 'likel'
    print gp.log_likel
    
    
    Xtest = np.tile(covariates, (n,1))
    Xtest = np.c_[magnitudes, Xtest]
    #Xtest = np.c_[magnitudes, magnitudes**2.0, Xtest]
    Xtest = np.array(Xtest, dtype=np.float)
    print Xtest
    Xtest = cov_filter.process(Xtest, reuse_stats=True)
    Xtest = np.c_[np.ones(len(Xtest))*-1, Xtest]
    yfit = gp.predict(Xtest)
    yfit = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
        
    np.savetxt('/home/marcel/datasets/multilevel/eusinan/bssa/comparison_paper/fig3/rjb10/gpsecorr_pred_pga.csv', np.c_[magnitudes,yfit], delimiter=',')
        
        
        
        
    
    
    