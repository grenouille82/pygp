'''
Created on Oct 29, 2012

@author: marcel
'''
import numpy as np
import upgeo.util.metric as metric
import upgeo.util.stats as stats

from upgeo.demo.util import loadmat_data
from upgeo.util.filter import StandardizeFilter, MeanShiftFilter
from upgeo.base.gp import GPRegression
from upgeo.base.infer import ExactInference
from upgeo.base.kernel import SEKernel, NoiseKernel, LinearKernel,\
    SqConstantKernel, ARDSEKernel, GroupNoiseKernel, HiddenKernel,\
    MaskedFeatureKernel, CorrelatedNoiseKernel

def create_noise_kernel(grp_idx, s, kernel=None, mask=None):
    noise_kernel = GroupNoiseKernel(grp_idx, s)
    if kernel != None:
        noise_kernel = HiddenKernel(noise_kernel)
        noise_kernel = noise_kernel*kernel
    if mask != None:
        noise_kernel = MaskedFeatureKernel(noise_kernel, mask)
    return noise_kernel

if __name__ == '__main__':
    #filename = '/home/marcel/datasets/multilevel/bssa_var_test/synthdata_a.mat'    
    #filename = '/home/marcel/datasets/multilevel/bssa_var_test/synthdata_a.mat'
    filename = '/home/marcel/datasets/multilevel/eusinan/bssa/eval_eudata_norm3a.mat'
    X,Y = loadmat_data(filename)

    n = X.shape[0]
    #choose target variable (see readme file fot the index coding)

    cov_filter = StandardizeFilter(1)
    target_filter = MeanShiftFilter()
    
    
    #Ynorm = np.squeeze(target_filter.process(Y))
    #Xnorm = cov_filter.process(X)
    Ynorm = Y[:,0]
    Xnorm = X
    
    fmask = np.r_[0, np.ones(7)]
    fmask = np.array(fmask, dtype=np.bool)
    
    
    #l = (np.max(Xnorm,0)-np.min(Xnorm,0))/2.0
    l = (np.max(Xnorm[:,fmask],0)-np.min(Xnorm[:,fmask],0))/2.0
    l[l == 0] = 1e-4
    print 'l={0}'.format(l)
    
    #standard kernels
    kernel = SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    #kernel = SEKernel(np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(l), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel() + NoiseKernel(np.log(0.5))
    #kernel = ARDSEKernel(np.log(l), np.log(1)) + ARDLinearKernel(np.log(1)*np.ones(len(l)), np.log(1)) + NoiseKernel(np.log(0.5))
    #kernel = ARDSELinKernel(np.log(l), np.log(1), np.log(1)) + NoiseKernel(np.log(0.5))

    #group noise kernel
    noise_kernel = create_noise_kernel(0, np.log(1))
    kernel = MaskedFeatureKernel(kernel, fmask) + noise_kernel
    
    #correlated noise kernel
    #kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask) + CorrelatedNoiseKernel(0, np.log(0.1), np.log(0.5))
    #kernel = MaskedFeatureKernel(ARDSEKernel(np.log(l), np.log(1)) + SqConstantKernel(np.log(1)) * LinearKernel(), fmask) + CorrelatedNoiseKernel(0, np.log(0.1), np.log(0.5))
        
    
    
    algo = GPRegression(kernel, infer_method=ExactInference)
    
    algo.fit(Xnorm, Ynorm)
    Yfit, var = algo.predict(Xnorm, ret_var=True)
    
    mse = metric.mspe(Ynorm, Yfit)
    nmse = mse/np.var(Ynorm)
    mll = metric.nlp(Ynorm, Yfit, var)
    nmll = mll-metric.nlpp(Ynorm, np.mean(Ynorm), np.var(Ynorm))

    
    print 'Training Errors:'
    print 'mse={0}, nmse={1}, mll={2}, nmll={3}'.format(mse, nmse, mll, nmll)
    print 'likel={0}'.format(algo.log_likel)
    print 'Hyperparameters:'
    print kernel.params 
    
    
    
    
    
    
    
    

