'''
Created on May 30, 2013

@author: marcel
'''
import numpy as np

from upgeo.demo.util import loadmat_data
from upgeo.util.filter import CompositeFilter, FunctionFilter, MinMaxFilter,\
    MeanShiftFilter
from upgeo.base.kernel import SEKernel, SqConstantKernel, LinearKernel,\
    NoiseKernel
from upgeo.base.infer import ExactInference
from upgeo.base.gp import GPRegression
from upgeo.regression.bayes import EMBayesRegression

if __name__ == '__main__':
    filename = '/home/marcel/datasets/multilevel/kappa/eval_kappa.mat'
    
    Xtrain,ytrain = loadmat_data(filename)
    Xtrain = Xtrain[:,np.newaxis]

    Xtest = np.arange(10,3500,10)
    Xtest = Xtest[:,np.newaxis]

    cov_filter = CompositeFilter([FunctionFilter(np.log, np.exp), MinMaxFilter()])
    #cov_filter = MinMaxFilter()
    #target_filter = CompositeFilter([FunctionFilter(np.log, np.exp), MeanShiftFilter()])
    target_filter = MeanShiftFilter()

    Xtrain = cov_filter.process(Xtrain)
    ytrain = np.squeeze(target_filter.process(ytrain[:,np.newaxis]))
    Xtest = cov_filter.process(Xtest,True)
    
    
    kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5)) 
    
    model = EMBayesRegression(alpha0=1, beta0=1, weight_bias=True)
    #model = GPRegression(kernel, infer_method=ExactInference)
    model.fit(Xtrain, ytrain)
    
    #vs30scale
    yfit, var = model.predict(Xtest, True)
    yfit = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
    #yfit = np.log(yfit)
    np.savetxt('/home/marcel/datasets/multilevel/kappa/viz/LogVS30Kappa/pooled/linreg.csv', np.c_[cov_filter.invprocess(Xtest),yfit,var], delimiter=',')
    
    #residuals
    yfit = model.predict(Xtrain, False)
    #resid = yfit-ytrain
    yfit = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
    #yfit = np.log(yfit)
    np.savetxt('/home/marcel/datasets/multilevel/kappa/viz/LogVS30Kappa/pooled/resid/linreg.csv', np.c_[cov_filter.invprocess(Xtrain),np.squeeze(target_filter.invprocess(ytrain[:,np.newaxis])), yfit], delimiter=',')
    
    print 'loglikel={0}'.format(model.log_likel)
    print 'hyperparams={0}'.format(np.exp(model.hyperparams))
    
    
    