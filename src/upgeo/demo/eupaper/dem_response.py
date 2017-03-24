'''
Created on Dec 2, 2012

@author: mhermkes
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
    train_fname = '/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/eudata_train_fullperiods.csv'
    
    pred_cols = np.arange(0,9)
    target_cols = np.arange(9,73)

    periods = np.loadtxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/gp_periods.csv')
    periods = periods[1:] #including pga
    print 'periods'
    print periods


    n = len(periods)
    
    Xtrain,Ytrain = load_train_data(train_fname, pred_cols, target_cols)
    #ytrain = Ytrain[:,9] #pga
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
    Ytrain = target_filter.process(Ytrain[:,1:]) 
    #ytrain = np.squeeze(target_filter.process(ytrain[:,np.newaxis]))

    print event_bag.shape
    print Xtrain.shape

    #for corr noise model    
    Xtrain = np.c_[event_bag, Xtrain]


    models = [0]*n
    model_devs = np.zeros(n) 
    model_intra_devs = np.zeros(n)
    model_inter_devs = np.zeros(n)
    for i in xrange(n):
        kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() 
        noise_kernel = GroupNoiseKernel(0, np.log(0.5)) + NoiseKernel(np.log(0.5))
        kernel = MaskedFeatureKernel(kernel, fmask) + noise_kernel
        
        gp = GPRegression(kernel, infer_method=ExactInference)
        gp.fit(Xtrain, Ytrain[:,i])
        models[i] = gp
        model_devs[i] = np.sqrt(np.exp(2*gp.hyperparams[-2])+np.exp(2*gp.hyperparams[-1]))
        model_intra_devs[i] = np.exp(gp.hyperparams[-1])
        model_inter_devs[i] = np.exp(gp.hyperparams[-2])
        print Ytrain.shape
        print i
        print 'period={0}'.format(periods[i])
        print 'params={0}'.format(gp.hyperparams)
        
    
    #print 'mean'
    #print target_filter._mean

    '''
    generate data for figure 7/8
    '''
    vs30 = [270,490,760]
#   distances = [10,100]
    magnitudes = [4,5,6,7,8] 
    mag_idx = 0
    vs30_idx = 6
    #covariates = [7,15,0,1,0] #mw=7,fdepth=15,fm=ss
    #covariates = [6,15,0,1,0] #mw=7,fdepth=15,fm=ss
    covariates = [15,0,1,0,10] #fdepth=15,fm=ss, dist=10
    
    for v in vs30:
        for mw in magnitudes:
            xtest = np.r_[mw, covariates, v]
            xtest = np.array(xtest,dtype=np.float)
            xtest = xtest[np.newaxis,:]
            print xtest
            xtest = cov_filter.process(xtest, reuse_stats=True)
            print xtest
            xtest = np.c_[-1, xtest]
            yfit = np.empty((1,n))
            
            
            for i in xrange(n):
                gp = models[i]
                yfit[:,i] = gp.predict(xtest)
            #print 'bung'
            #print yfit
            yfit = np.squeeze(target_filter.invprocess(yfit))
            #print yfit
            
            np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig78/gpselincorr_mag{0}_vs{1}_pred_spectra.csv'.format(mw,v), np.c_[periods,yfit], delimiter=',')
            
       
    '''
    generate data for figure 4
    '''
    vs30 = [270,490,760]
    magnitudes = [4,5,6,7,8] 
    distances = [10,100]
    mag_idx = 0
    vs30_idx = 6
    #covariates = [7,15,0,1,0] #mw=7,fdepth=15,fm=ss
    #covariates = [6,15,0,1,0] #mw=7,fdepth=15,fm=ss
    covariates_rev = [15,1,0,0]
    covariates_ss = [15,0,1,0] #mw7, fdepth=15,fm=ss
    covariates_norm = [15,0,0,1]
 
    for v in vs30:
        for mw in magnitudes:
            for r in distances:
                xtest_rev = np.r_[mw, covariates_rev, r, v]
                xtest_ss = np.r_[mw, covariates_ss, r, v]
                xtest_norm = np.r_[mw, covariates_norm, r, v]
                
                xtest_rev = np.array(xtest_rev,dtype=np.float)
                xtest_ss = np.array(xtest_ss,dtype=np.float)
                xtest_norm = np.array(xtest_norm,dtype=np.float)
                
                xtest_rev = cov_filter.process(xtest_rev[np.newaxis,:], reuse_stats=True)
                xtest_ss = cov_filter.process(xtest_ss[np.newaxis,:], reuse_stats=True)
                xtest_norm = cov_filter.process(xtest_norm[np.newaxis,:], reuse_stats=True)
                
                xtest_rev = np.c_[-1, xtest_rev]
                xtest_ss = np.c_[-1, xtest_ss]
                xtest_norm = np.c_[-1, xtest_norm]
                
                yfit_rev = np.empty((1,n))
                yfit_ss = np.empty((1,n))
                yfit_norm = np.empty((1,n))
                
                
                for i in xrange(n):
                    gp = models[i]
                    yfit_rev[:,i] = gp.predict(xtest_rev)
                    yfit_ss[:,i] = gp.predict(xtest_ss)
                    yfit_norm[:,i] = gp.predict(xtest_norm)
                #print 'bung'
                #print yfit
                yfit_rev = np.squeeze(target_filter.invprocess(yfit_rev))
                yfit_ss = np.squeeze(target_filter.invprocess(yfit_ss))
                yfit_norm = np.squeeze(target_filter.invprocess(yfit_norm))
                #print yfit
                
                yfit_ratio_rev_ss = np.exp(yfit_rev-yfit_ss)
                yfit_ratio_norm_ss = np.exp(yfit_norm-yfit_ss)
                
                np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig4/gpselincorr_mag{0}_vs{1}_dist{2}_pred_spectra_strikeslip.csv'.format(mw,v,r), np.c_[periods,yfit_ss], delimiter=',')
                np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig4/gpselincorr_mag{0}_vs{1}_dist{2}_pred_spectra_reverse.csv'.format(mw,v,r), np.c_[periods,yfit_rev], delimiter=',')
                np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig4/gpselincorr_mag{0}_vs{1}_dist{2}_pred_spectra_normal.csv'.format(mw,v,r), np.c_[periods,yfit_norm], delimiter=',')
                np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig4/gpselincorr_mag{0}_vs{1}_dist{2}_spectra_ratio_reverse_strikeslip.csv'.format(mw,v,r), np.c_[periods,yfit_ratio_rev_ss], delimiter=',')
                np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig4/gpselincorr_mag{0}_vs{1}_dist{2}_spectra_ratio_normal_strikeslip.csv'.format(mw,v,r), np.c_[periods,yfit_ratio_norm_ss], delimiter=',')
                

    '''
    generate data for figure 6
    '''
    magnitudes = [4,5,6,7,8] 
    distances = [10,100]
    #covariates = [7,15,0,1,0] #mw=7,fdepth=15,fm=ss
    #covariates = [6,15,0,1,0] #mw=7,fdepth=15,fm=ss
    #covariates_rev = [15,1,0,0]
    covariates_vs270 = [15,0,1,0,270] #fdepth=15,fm=ss, vs=270
    covariates_vs490 = [15,0,1,0,490] #fdepth=15,fm=ss, vs=490
    covariates_vs760 = [15,0,1,0,760] #fdepth=15,fm=ss, vs=760
 
    for mw in magnitudes:
        for r in distances:
            xtest_vs270 = np.r_[mw, covariates_vs270[0:4], r, covariates_vs270[4]]
            xtest_vs490 = np.r_[mw, covariates_vs490[0:4], r, covariates_vs490[4]]
            xtest_vs760 = np.r_[mw, covariates_vs760[0:4], r, covariates_vs760[4]]
            
            print 'vs30 test'
            print xtest_vs760
            
            xtest_vs270 = np.array(xtest_vs270,dtype=np.float)
            xtest_vs490 = np.array(xtest_vs490,dtype=np.float)
            xtest_vs760 = np.array(xtest_vs760,dtype=np.float)
            
            xtest_vs270 = cov_filter.process(xtest_vs270[np.newaxis,:], reuse_stats=True)
            xtest_vs490 = cov_filter.process(xtest_vs490[np.newaxis,:], reuse_stats=True)
            xtest_vs760 = cov_filter.process(xtest_vs760[np.newaxis,:], reuse_stats=True)
            
            xtest_vs270 = np.c_[-1, xtest_vs270]
            xtest_vs490 = np.c_[-1, xtest_vs490]
            xtest_vs760 = np.c_[-1, xtest_vs760]
            
            yfit_vs270 = np.empty((1,n))
            yfit_vs490 = np.empty((1,n))
            yfit_vs760 = np.empty((1,n))
            
            
            for i in xrange(n):
                gp = models[i]
                yfit_vs270[:,i] = gp.predict(xtest_vs270)
                yfit_vs490[:,i] = gp.predict(xtest_vs490)
                yfit_vs760[:,i] = gp.predict(xtest_vs760)
            #print 'bung'
            #print yfit
            yfit_vs270 = np.squeeze(target_filter.invprocess(yfit_vs270))
            yfit_vs490 = np.squeeze(target_filter.invprocess(yfit_vs490))
            yfit_vs760 = np.squeeze(target_filter.invprocess(yfit_vs760))
            #print yfit
            
            yfit_ratio_vs270_vs760 = np.exp(yfit_vs270-yfit_vs760)
            yfit_ratio_vs490_vs760 = np.exp(yfit_vs490-yfit_vs760)
            
            np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig6/gpselincorr_mag{0}_vs270_dist{1}_pred_spectra.csv'.format(mw,r), np.c_[periods,yfit_vs270], delimiter=',')
            np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig6/gpselincorr_mag{0}_vs490_dist{1}_pred_spectra.csv'.format(mw,r), np.c_[periods,yfit_vs490], delimiter=',')
            np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig6/gpselincorr_mag{0}_vs760_dist{1}_pred_spectra.csv'.format(mw,r), np.c_[periods,yfit_vs760], delimiter=',')
            np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig6/gpselincorr_mag{0}_dist{1}_spectra_ratio_vs270_vs760.csv'.format(mw,r), np.c_[periods,yfit_ratio_vs270_vs760], delimiter=',')
            np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig6/gpselincorr_mag{0}_dist{1}_spectra_ratio_vs490_vs760.csv'.format(mw,r), np.c_[periods,yfit_ratio_vs490_vs760], delimiter=',')

        
            
    '''
    generate data for figure 8
    '''
    np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig9/gpselincorr_model_stddevs.csv', np.c_[periods,model_devs], delimiter=',')
    np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig9/gpselincorr_model_within_stddevs.csv', np.c_[periods,model_intra_devs], delimiter=',')
    np.savetxt('/home/mhermkes/datasets/multilevel/eusinan/bssa/comparison_paper/fig9/gpselincorr_model_between_stddevs.csv', np.c_[periods,model_inter_devs], delimiter=',')
        
            
        
        
        
    
    
