'''
Created on May 30, 2013

@author: marcel
'''
import numpy as np


from upgeo.util.array import unique
from upgeo.demo.util import loadmat_data, loadmat_mtl_data
from upgeo.util.filter import CompositeFilter, FunctionFilter, MinMaxFilter,\
    MeanShiftFilter
from upgeo.base.kernel import SEKernel, SqConstantKernel, LinearKernel,\
    NoiseKernel
from upgeo.base.infer import ExactInference
from upgeo.base.gp import GPRegression
from upgeo.regression.bayes import EMBayesRegression

if __name__ == '__main__':
    filename = '/home/marcel/datasets/multilevel/kappa/eval_kappa_mtl.mat'
    
    Xtrain,ytrain,tasks = loadmat_mtl_data(filename)
    Xtrain = Xtrain[:,np.newaxis]
    itasks = unique(tasks)
    k = len(itasks)
    n = Xtrain.shape[0]
    
    
    Xtest = np.arange(10,3500,10)
    Xtest = Xtest[:,np.newaxis]

    cov_filter = CompositeFilter([FunctionFilter(np.log, np.exp), MinMaxFilter()])
    #cov_filter = MinMaxFilter()
    #target_filter = CompositeFilter([FunctionFilter(np.log, np.exp), MeanShiftFilter()])
    target_filter = MeanShiftFilter()

    Xtrain = cov_filter.process(Xtrain)
    ytrain = np.squeeze(target_filter.process(ytrain[:,np.newaxis]))
    Xtest = cov_filter.process(Xtest,True)
    
    
    #kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5)) 
    
    model = EMBayesRegression(alpha0=1, beta0=1, weight_bias=True)
    #model = GPRegression(kernel, infer_method=ExactInference)
    
    params = np.copy(model.hyperparams)
    Yfit = np.zeros(n)
    for i in xrange(k):
        train_ids = tasks == itasks[i]
        
        model.hyperparams = params
        model.fit(Xtrain[train_ids], ytrain[train_ids])
        
        yfit, var = model.predict(Xtest, ret_var=True)
            
        #vs30scale
        yfit, var = model.predict(Xtest, True)
        yfit = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
        #yfit = np.log(yfit)
        np.savetxt('/home/marcel/datasets/multilevel/kappa/viz/LogVS30Kappa/stl/linreg_task{0}.csv'.format(itasks[i]), np.c_[cov_filter.invprocess(Xtest),yfit,var], delimiter=',')
    
        #residuals
        Yfit[train_ids] = model.predict(Xtrain[train_ids], False)
        #resid = yfit-ytrain
        
        print 'Task Model {0}:'.format(itasks[i])
        print 'loglikel={0}'.format(model.log_likel)
        print 'hyperparams={0}'.format(np.exp(model.hyperparams))
        
    Yfit = np.squeeze(target_filter.invprocess(Yfit[:,np.newaxis]))
    #Yfit = np.log(Yfit)
    np.savetxt('/home/marcel/datasets/multilevel/kappa/viz/LogVS30Kappa/stl/resid/linreg.csv', np.c_[cov_filter.invprocess(Xtrain),np.squeeze(target_filter.invprocess(ytrain[:,np.newaxis])), Yfit], delimiter=',')
    
        
    
    
