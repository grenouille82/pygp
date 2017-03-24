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
    NoiseKernel, ExpGaussianKernel, DiracConvolvedKernel, FixedParameterKernel
from upgeo.base.infer import ExactInference
from upgeo.base.gp import GPRegression
from upgeo.regression.bayes import EMBayesRegression
from upgeo.mtl.kernel import ConvolvedMTLKernel
from upgeo.mtl.infer import CMOGPExactInference
from upgeo.mtl.gp import CMOGPRegression

if __name__ == '__main__':
    filename = '/home/marcel/datasets/multilevel/kappa/eval_kappa_mtl.mat'
    
    Xtrain,ytrain,tasks = loadmat_mtl_data(filename)
    Xtrain = Xtrain[:,np.newaxis]
    task_ids, itasks = unique(tasks,True)
    k = len(itasks)
    n = Xtrain.shape[0]
    
    
    Xtest = np.arange(10,3500,10)
    Xtest = Xtest[:,np.newaxis]

    #cov_filter = CompositeFilter([FunctionFilter(np.log, np.exp), MinMaxFilter()])
    cov_filter = MinMaxFilter()
    #target_filter = CompositeFilter([FunctionFilter(np.log, np.exp), MeanShiftFilter()])
    target_filter = MeanShiftFilter()

    Xtrain = cov_filter.process(Xtrain)
    ytrain = np.squeeze(target_filter.process(ytrain[:,np.newaxis]))
    Xtest = cov_filter.process(Xtest,True)
    
    #latent_kernel = ExpGaussianKernel(np.log(0.1))
    #latent_kernel = CompoundKernel([ExpGaussianKernel(np.log(0.1)), ExpGaussianKernel(np.log(0.2))])
    latent_kernel = DiracConvolvedKernel(FixedParameterKernel(SEKernel(np.log(0.1),np.log(1)), [1]))
    noise_kernel = SEKernel(np.log(0.1), np.log(0.1)) + NoiseKernel(np.log(0.1))
    #noise_kernel = SEKernel(np.log(0.1), np.log(0.1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.1))
    
    #theta = [np.log(0.1), np.log(0.1)]
    #theta = [np.log(0.1), np.log(0.1), np.log(0.2), np.log(0.1)]
    theta = [np.log(0.1)]
    kernel = ConvolvedMTLKernel(latent_kernel, theta, k, noise_kernel) 

    model = CMOGPRegression(kernel, infer_method=CMOGPExactInference)
    model.fit(Xtrain, ytrain, itasks)

    Yfit = np.zeros(n)
    for i in xrange(k):
        train_ids = tasks == task_ids[i]
            
        #vs30scale
        yfit, var = model.predict_task(Xtest, q=i, ret_var=True)
        yfit = np.squeeze(target_filter.invprocess(yfit[:,np.newaxis]))
        #yfit = np.log(yfit)
        np.savetxt('/home/marcel/datasets/multilevel/kappa/viz/VS30Kappa/mtl/dirac_gpse_task{0}.csv'.format(task_ids[i]), np.c_[cov_filter.invprocess(Xtest),yfit,var], delimiter=',')
    
        #residuals
        Yfit[train_ids] = model.predict_task(Xtrain[train_ids], q=i, ret_var=False)
        #resid = yfit-ytrain
        
        print 'Task Model {0}:'.format(itasks[i])
        print 'loglikel={0}'.format(model.log_likel)
        print 'hyperparams={0}'.format(np.exp(model.hyperparams))
        
    Yfit = np.squeeze(target_filter.invprocess(Yfit[:,np.newaxis]))
    #Yfit = np.log(Yfit)
    np.savetxt('/home/marcel/datasets/multilevel/kappa/viz/VS30Kappa/mtl/resid/dirac_gpse.csv', np.c_[cov_filter.invprocess(Xtrain),np.squeeze(target_filter.invprocess(ytrain[:,np.newaxis])), Yfit], delimiter=',')
    
        
    
    
