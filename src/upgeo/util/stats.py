'''
Created on May 4, 2012

@author: marcel
'''

import numpy as np
import numpy.linalg as lin
import scipy.special as sps
from scipy.linalg.decomp_cholesky import cho_solve

def normpdf(x, mu=0, sigma=1):
    '''
    '''
    x_center = x-mu
    pdf = 1.0/np.sqrt(2.0*np.pi*sigma) * np.exp(-(x_center**2)/(2.0*sigma))
    return pdf

def normlnpdf(x, mu=0, sigma=1):
    '''
    '''
    x_center = x-mu
    pdf = -0.5*np.log(2.0*np.pi*sigma) - (x_center**2)/(2.0*sigma)
    return pdf

def gradient_normpdf(x, mu=0, sigma=1):
    '''
    '''
    pdf = normpdf(x, mu, sigma)
    dlnmu, dlnsigma = gradient_normlnpdf(x, mu, sigma)
    dmu = pdf * dlnmu
    dsigma = pdf * dlnsigma
    return dmu, dsigma 

def gradient_normlnpdf(x, mu=0, sigma=1):
    '''
    '''
    x_center = x-mu
    
    dmu = x_center/sigma
    dsigma = -0.5/sigma + (x_center**2)/(2.0*sigma**2)
    return dmu, dsigma

def mvnpdf(x, mu=None, sigma=None):
    '''
    
    TODO: - for performance reasons distinguish between a spherical, diagonal and full 
            covariance matrix.
          - error handling
    '''
    return np.exp(mvnlnpdf(x, mu, sigma))
    
def mvnlnpdf(x, mu=None, sigma=None):  
    '''
    
    TODO: - for performance reasons distinguish between a spherical, diagonal and full 
            covariance matrix.
          - error handling
    '''
    x = np.atleast_2d(x)
    d = x.shape[1]
    
    mu = np.zeros(d) if mu is None else np.asarray(mu)
    sigma = np.identity(d) if sigma is None else np.asarray(sigma)
    
    #assert sigma.ndim == 2 and sigma.shape[0] == sigma.shape[1], \
    #("sigma should be a square matrix with same dimension as x")  
    #assert mu.ndim == 1 and mu.size == d, ("mu should be a vector with same dimension as x")
    
    #center data around the mean
    x_center = x - mu
    
    if sigma.ndim == 2:
    
        #calc the quadratic form by the inner product of standardized form
        R = lin.cholesky(sigma).T 
        x_Rinv = np.dot(x_center, lin.inv(R)) 
        
        #log determinant of sigma
        det_sigma = np.sum(np.log(np.diag(R)))
        
    else:
        n = len(sigma)
        x_Rinv = np.zeros(n)
        det_sigma = np.zeros(n)
        for i in xrange(n):
            #calc the quadratic form by the inner product of standardized form
            R = lin.cholesky(sigma).T
            x_Rinv[i] = np.dot(x_center, lin.inv(R))
            
            #log determinant of sigma
            det_sigma[i] = np.sum(np.log(np.diag(R)))
            
    qf = np.sum(x_Rinv**2, 1)
    y = -0.5*qf - det_sigma - d*np.log(2*np.pi)/2
    return y

def mvtpdf(x, mu, sigma, df):
    return np.exp(mvtlnpdf(x, mu, sigma, df))

def gradient_mvtpdf(x, mu, sigma, df):
    pdf = mvtpdf(x, mu, sigma, df)
    dlnmu, dlnsigma, dldf = gradient_mvtlnpdf(x, mu, sigma, df)
    dmu = pdf * dlnmu
    dsigma = pdf * dlnsigma
    ddf = pdf * dldf
    return dmu, dsigma, ddf 


def gradient_mvtlnpdf(x, mu, sigma, df):
    '''
    '''
    x = np.atleast_2d(x)
    d = x.shape[1]
    
    
    x_center = x-mu
    if sigma.ndim == 2:
        R = np.linalg.cholesky(sigma).T
        detS = 2.0*np.sum(np.log(np.diag(R)))
        x_Rinv = np.dot(x_center, np.linalg.inv(R))
        print R.shape
        print x_center.shape
        x_Sinv = cho_solve((R, 0), x_center.T)
        invS = cho_solve((R, 0), np.eye(d))
    else:
        n = len(sigma)
        x_Rinv = np.zeros(n)
        x_Sinv = np.zeros(n)
        invS = np.zeros(n)
        detS = np.zeros(n)
        for i in xrange(n):
            #calc the quadratic form by the inner product of standardized form
            R = lin.cholesky(sigma[i]).T
            x_Rinv[i] = np.dot(x_center, lin.inv(R))
            x_Sinv[i] = cho_solve((R, False), x_center)
            invS[i] = cho_solve((R, 0), np.eye(d))
            #log determinant of sigma
            detS[i] = 2.0*np.sum(np.log(np.diag(R)))


    qf = np.sum(x_Rinv**2, 1)
    dmu = 2.0*x_Sinv*((d+df)/2.0) / (df*(qf/df + 1.0))
    ddf = 0.5*(sps.digamma(df/2.0+d/2.0) - sps.digamma(df/2.0)) - d/(2.0*df) 
    ddf += (qf * (d+df)/2.0) / (df**2 * (qf/df + 1.0)) - np.log(qf/df+1)/2.0
    dsigma = -0.5*invS + np.outer(x_center, x_center)*((d+df)/2.0) / (df*(qf/df + 1.0))
    dsigma = (dsigma+dsigma.T)#/2.0
    
    return dmu, dsigma, ddf

    
def mvtlnpdf(x, mu, sigma, df):
    ''' 
    '''
    x = np.atleast_2d(x)
    d = x.shape[1]
    
    
    x_center = x-mu
    
    if sigma.ndim == 2:
        R = np.linalg.cholesky(sigma).T
        detS = 2.0*np.sum(np.log(np.diag(R)))
        x_Rinv = np.dot(x_center, np.linalg.inv(R))
    else:
        '''
        '''
        n = len(sigma)
        x_Rinv = np.zeros(n)
        detS = np.zeros(n)
        for i in xrange(n):
            #calc the quadratic form by the inner product of standardized form
            R = lin.cholesky(sigma[i]).T
            x_Rinv[i] = np.dot(x_center, lin.inv(R))
            
            #log determinant of sigma
            detS[i] = 2.0*np.sum(np.log(np.diag(R)))
    
    #term A = 
    A = sps.gammaln(df/2.0 + d/2.0) - sps.gammaln(df/2.0)
    
    #term B =
    B = (-detS - d*np.log(df) - d*np.log(np.pi)) / 2.0
    
    #term C =  
    qf = np.sum(x_Rinv**2, 1)
    C = -(df+d)/2.0 * np.log(1.0+qf/df)
    
    pdf = A+B+C
    pdf = np.squeeze(pdf)
    return pdf

def tpdf(x, mu=0, sigma=1, df=2):
    '''
    '''
    x_center = x-mu
    c = np.exp(sps.gammaln(df/2.0 + 0.5) - sps.gammaln(df/2.0)) * 1.0/np.sqrt(df*np.pi*sigma)
    pdf = c * (1.0 + 1.0/df * (x_center**2/sigma))**(-(df+1.0)/2.0)
    return pdf

def tlnpdf(x, mu=0, sigma=1, df=2):
    x_center = x-mu
    c = sps.gammaln(df/2.0 + 0.5) - sps.gammaln(df/2.0) - 0.5 * np.log(np.pi*sigma*df)
    pdf = c - (df+1.0)/2.0 * np.log(1.0 + 1.0/df * (x_center**2/sigma))
    return pdf

def gradient_tpdf(x, mu=0, sigma=1, df=2):
    '''
    '''
    pdf = tpdf(x, mu, sigma, df)
    dlnmu, dlnsigma, dldf = gradient_tlnpdf(x, mu, sigma, df)
    dmu = pdf * dlnmu
    dsigma = pdf * dlnsigma
    ddf = pdf * dldf
    return dmu, dsigma, ddf 
    
def gradient_tlnpdf(x, mu=0, sigma=1, df=2):
    x_center = x-mu
    x_sqcenter = x_center**2
    x_std = x_sqcenter/sigma
    
    dmu = (df+1.0) * x_center/(df*sigma*(x_std/df+1.0))
    dsigma = -0.5*(1.0/sigma) + (df+1.0)/2.0 *  x_sqcenter/(sigma**2 * df * (x_std/df + 1.0))
    ddf = 0.5 * (sps.digamma(df/2.0 + 0.5) - sps.digamma(df/2.0) - 1.0/df)
    ddf += (df+1.0)/2.0 * x_sqcenter / (df**2 * sigma * (x_std/df + 1.0))
    ddf -= np.log(x_std/df + 1.0)/2.0
    
    return dmu, dsigma, ddf

def nanmean(x, axis=None, dtype=None):
    '''
    '''
    x, axis = _chk_asarray(x, axis, dtype);
    x = x.copy();
    
    nan_mask = np.isnan(x);
    
    n_orig = x.shape[axis];
    n_nan = np.sum(np.isnan(x), axis)*1.0
    n = n_orig - n_nan
    
    np.putmask(x, nan_mask, 0)
    mean = np.sum(x, axis) / n
    
    return mean;
    
def nanvar(x, axis=None, dtype=None, dof=1):
    '''
    '''
    assert dof >= 0
    
    x, axis = _chk_asarray(x, axis, dtype);
    x = x.copy();
    
    nan_mask = np.isnan(x);
    
    n_orig = x.shape[axis];
    n_nan = np.sum(np.isnan(x), axis)*1.0
    n = n_orig - n_nan
    
    assert n_orig > dof, "number of cases must be larger than dof"
    
    n -= dof
    
    np.putmask(x, nan_mask, 0)
    mean = np.sum(x, axis) / n
    
    if axis is not None:
        x_center = x-np.expand_dims(mean, axis)
    else:
        x_center = x-mean
        
    var = np.sum(x_center**2) / n 
    return var;

def nanstd(x, axis=None, dtype=None, dof=1):
    '''
    '''    
    var = nanvar(x, axis, dtype, dof)
    std = np.sqrt(var)
    return std

def nancov(x, y=None, dtype=None, dof=1, ctype='cc'):
    '''
    TODO: - check if the bias parameter was set correctly by invoking np.cov(...)
          - implement pairwise calculation
    '''
    if y is not None:
        x = _chk_asarray(x, None, dtype)
        y = _chk_asarray(y, None, dtype)
        x = np.vstack((x,y)).transpose()
        
    assert x.ndim == 2, 'x must be a square matrix'
    
    if ctype == 'cc':
        #determine covariance matrix by considering complete cases
        compl_cases = x[:,~np.any(np.isnan(x), 0)]
        cov = np.cov(compl_cases, bias=1-dof);
    elif ctype == 'pw':
        #determine covariance matrix by pairwise computation
        raise NotImplementedError("method with pairwise computation is not supported yet")
    else:
        raise TypeError("bad ctype: '{0}'".format(ctype))
    
    return cov;

def covcorr(S):
    assert S.ndim == 2 and S.shape[0]== S.shape[1], 'S must be a square matrix'

    D = np.sqrt(np.diag(S))
    iD = 1/D
    iD = np.diag(iD)
    R = np.dot(np.dot(iD, S), iD)
    return R

def corrcov(R, var):
    assert R.ndim == 2 and R.shape[0]== R.shape[1], 'R must be a square matrix'
    assert var.ndim == 1 and var.shape[0] == R.shape[0], 'var must be a vector of variances'
    
    D = np.diag(var)
    S = np.dot(np.dot(D, R), D)
    return S
    

#def mode(x, window=1, axis=0):
#    """
#    Calculate the mode for continuous data in array x see Numerical Recipes, 
#    Chapter 13
#
#    usage:  index_list, probability_list = mode(array_of_data,window)
#    returns two lists: 
#    1) the index {i.e. the value from the data calculated as (x[i]+x[i+window])/2}
#    2) the probability of finding that value
#
#
#    """
#    # make sure data is in an array and make sure it is sorted 
#    # (will not maintain synchronicity between columns though, but that shouldn't matter
#    # for the mode calculation!)
#    x = np.asarray(x)
#    x = np.msort(x)
#    
#    # create the index array
#    ind = N.zeros((len(x)-j,x.shape[1]),float)
#    # create the probability array
#
#    p = N.zeros((len(x)-j,x.shape[1]),float)
#    n=len(x)
#    for i in range(n-j):
#        ind[i] = N.multiply(0.5,add(x[i],x[i+j]))
#        p[i] = N.divide(j,N.multiply(n,N.subtract(x[i+j],x[i])))
#        return ind, p


##############################
### Descriptive Statistics ###
##############################

def weighttest(x,w):
    """Checks and formats weights.
    
    Allows the use of uncertianties using ErrorVal as the weights for the
    statistical functions.  This function also makes sure that nan values in
    the data don't have weights associated with them as that would throw off
    the calculation of the statistics when nan values are ignored.
    
    Parameters:
        x : ndarray or ArrayOfErr
            The data that the weights are supposed to correspond to.
        w : ndarray
            The weight set.  If None, we check for the use of ErrorVal first.
            If ErrorVal is used, weights are assigned as the inverse of the
            total uncertianty for each data point.  If ErrorVal is not used,
            then None defaults to equal weighting.
    Returns:
        y : ndarray
            The data set ready for the statistics functions to use.
        w : ndarray
            The weight set ready for the statistics functions to use.
    See Also:
        ErrorVal (available at http://users.bigpond.net.au/gazzar/python.html)
    """
    if w == None:
        try:
            from ErrorVal import NegErrs,PosErrs,PrimeVals
            w = 1/(NegErrs(x)+PosErrs(x))
            y = PrimeVals(x)
        except (ImportError, AttributeError):
            w = np.ones_like(x)
            y = x
    else:
        try:
            from ErrorVal import PrimeVals
            y = PrimeVals(x)
        except (ImportError, AttributeError):
            y = x
    w = w*np.abs(np.isnan(y)-1) #makes sure that nan data points don't have a weight
    return np.array(y),w

def absmean(x,w=None,axis=None,NN=True):
    """Computes the algebreic mean of the absolute values of the input array.
    
    Parameters:
        x : ndarray or ArrayOfErr
            The data which will be averaged.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the absolute mean is to be taken.  If none is
            given then the absolute mean will be taken over the entire array.
    Returns:
        result : float
            The algebreic mean of the absolute values of x.
    """
    x,w = weighttest(x,w)
    if NN:
        result = 1.*np.sum(np.abs(x*w),axis=axis)/np.sum(w,axis=axis)
    else:
        result = 1.*np.nansum(np.abs(x*w),axis=axis)/np.nansum(w,axis=axis)
    return result

def geomean(x,w=None,axis=None,NN=True):
    """Computes the geometric mean of the input array.
    
    Parameters:
        x : ndarray or ArrayOfErr
            The data which will be averaged.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the geometric mean is to be taken.  If none is
            given then the geometric mean will be taken over the entire array.
    Returns:
        result : float
            The geometric mean of x.
    """
    x,w = weighttest(x,w)
    if NN:
        result = np.product(x**w,axis=axis)**(1./np.sum(w,axis=axis))
    else:
        result = 1.
        for i in range(len(x)):
            if x[i] == np.nan or w[i] == np.nan:
                continue
            else:
                result *= x[i]**w[i]
        result = result**(1./np.nansum(w,axis=axis))
    return result

def harmean(x,w=None,axis=None,NN=True):
    """Computes the harmonic mean of the input array.
    
    Parameters:
        x : ndarray or ArrayOfErr
            The data which will be averaged.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the harmonic mean is to be taken.  If none is
            given then the harmonic mean will be taken over the entire array.
    Returns:
        result : float
            The harmonic mean of x.
    """
    x,w = weighttest(x,w)
    if any(x == 0):
        return np.nan
    if NN:
        result = 1.*np.sum(w,axis=axis)/np.sum(w/x,axis=axis)
    else:
        result = 1.*np.nansum(w,axis=axis)/np.nansum(w/x,axis=axis)
    return result

def quadmean(x,w=None,axis=None,NN=True):
    """Computes the quadratic mean of the input array.
        
    Parameters:
        x : ndarray or ArrayOfErr
            The data which will be averaged.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the quadratic mean is to be taken.  If none is
            given then the quadratic mean will be taken over the entire array.
    Returns:
        result : float
            The quadratic mean of x.
    """
    x,w = weighttest(x,w)
    if NN:
        result = np.sqrt(1.*np.sum(w*x**2,axis=axis)/np.sum(w,axis=axis))
    else:
        result = np.sqrt(1.*np.nansum(w*x**2,axis=axis)/np.nansum(w,axis=axis))
    return result

def mean(x,w=None,axis=None,NN=True):
    """Computes the mean.
            
    Parameters:
        x : ndarray or ArrayOfErr
            The data which will be averaged.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        axis : integer
            The axis over which the mean is to be taken.  If none is given then
            the mean will be taken over the entire array.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
    Returns:
        result : float
            The mean of x.
    """
    x,w = weighttest(x,w)
    if NN:
        result = 1.*np.sum(x*w,axis=axis)/np.sum(w)
    else:
        result = 1.*np.nansum(x*w,axis=axis)/np.nansum(w)
    return result

def median(x,w=None,axis=None,NN=True):
    """Calculates the median (middle value).
    
    Interface level function that provides axis control.  See source of
    median_work for actual computation of the median.
    
    Parameters:
        x : ndarray or ArrayOfErr
            The data which will be averaged.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        axis : integer
            The axis over which the median is to be taken.  If none is given
            then the median will be taken over the entire array.
        NN : boolean
            If True (default) nan values in x and w will not be ignored and so 
            nan will be returned if they are present.  If False, then nan 
            values will be ignored in x and weights of nan will be treated as a
            weight of 0.
    
    Returns:
        result : float or ndarray
            The median of x.
    """
    x,w = weighttest(x,w)
    mytype = [('data',x.dtype),('weight',w.dtype)]
    d = np.zeros(np.shape(x),dtype=mytype)
    d['data'] = x
    d['weight'] = w
    if axis == None:
        result = median_work(d,NN)
    else:
        result = np.apply_along_axis(median_work,axis,d,NN)
    return result

def median_work(d,NN=True):
    """Calculates the median (middle value).
    
    Parameters:
        d : ndarray
            The data and weight corresponding to each data point in a single 
            rank 1 array.
        NN : boolean
            If True (default) nan values in x and w will not be ignored and so 
            nan will be returned if they are present.  If False, then nan 
            values will be ignored in x and weights of nan will be treated as a
            weight of 0.
    Returns:
        result : float
            The median of the data in d.
    """
    x,w = d['data'],d['weight']
    if (np.any(np.isnan(x)) or np.any(np.isnan(w))) and NN:
        return np.nan
    t = np.nansum(w)/2.
    xrankable = []
    missing = np.isnan(x)
    for i in range(len(x.flat)):
        if not missing.flat[i] and w.flat[i] != 0:
            xrankable.append([x.flat[i],w.flat[i]])
    xrankable.sort()
    if len(xrankable) == 0:
        result = np.nan
    elif len(xrankable) == 1:
        result = float(xrankable[0][0])
    else:
        cumw = np.cumsum(np.array(xrankable)[:,1])
        for i in range(len(cumw)):
            if cumw[i-1] < t < cumw[i]:
                result = float(xrankable[i][0])
                break
            elif cumw[i] == t:
                result = (xrankable[i][0]+xrankable[i+1][0])/2.
                break
    return result

def mode(x,w=None,NN=True):
    """Finds all modes (most frequent values) of an input array.
    
    Because the mode is well defined even when some data is nan, this function
    does not use weighttest (which would set the weights of nan to nan) but
    instead does its own testing of the weights.
    Because the number of modes can vary for each data set, the results do not
    lend themselves to being held in an array when attempting to apply mode
    along a particular axis.  For example, the modes along axis=1 for 
    array([[1,1,2,3],[1,1,2,2],[1,2,3,4]]) are [1], [1,2], and 
    [1,2,3,4].  This information cannot be held in a single ndarray because of
    the variable number of modes.
    
    Parameters:
        x : ndarray or ArrayOfErr
            The data which will be averaged.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in w will not be ignored and so nan
            will be returned if they are present.  If False, then weights of 
            nan will be treated as a weight of 0.
    Returns:
        result : ndarray
            A rank 1 array containing all the modes of x.
    """
    if w == None:
        try:
            from ErrorVal import NegErrs,PosErrs,PrimeVals
            w = 1/(NegErrs(x)+PosErrs(x))
            x = PrimeVals(x)
        except (ImportError, AttributeError):
            w = np.ones_like(x)
    else:
        try:
            from ErrorVal import PrimeVals
            x = PrimeVals(x)
        except (ImportError, AttributeError):
            x = x    
    from scipy import unique
    y = unique(x.flat)
    count = np.zeros_like(y)
    for i in range(len(y)):
        for j in range(len(x.flat)):
            if x.flat[j] == y[i]:
                count[i] += w.flat[j]
    m = []
    if any(np.isnan(count)) and NN:
        return np.nan
    for i in range(len(y)):
        if count[i] == np.nanmax(count):
            m.append(y[i])
    return np.array(m)

def moment(x,n,w=None,axis=None,NN=True):
    """Calculates a moment of the data.
    
    Finds the nth moment (sum of (the deviations to the nth power)) of the 
    input array.  Used primarily to simplify other functions.
    Since this is not a normalized statistic use of non-integer weights should
    be carefully considered prior to use.
    
    Parameters:
        x : ndarray or ArrayOfErr
            The data for which the moment will be caluclated.
        n : float
            The order of the moment to be calculated.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the moment is to be taken.  If none is given
            then the moment will be taken over the entire array.
    Returns:
        result : float
            The nth moment of x.
    """
    x,w = weighttest(x,w)
    if NN:
        result = np.sum(w*(x - mean(x,w,axis,NN))**n,axis=axis)
    else:
        result = np.nansum(w*(x - mean(x,w,axis,NN))**n,axis=axis)
    return result

def sumsqrdev(x,w=None,axis=None,NN=True):
    """Finds the sum of the squared deviations of an input array.
    
    Equivalent to the 2nd moment of the data.
    Since this is not a normalized statistic use of non-integer weights should
    be carefully considered prior to use.
    
    Parameters:
        x : ndarray or ArrayOfErr
            The data set.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the sum of the squared deviations is to be 
            taken.  If none is given then the sum of the squared deviations 
            will be taken over the entire array.
    Returns:
        result : float
            The sum of squared deviations.
    See Also:
        moment
    """
    result = moment(x,2,w,axis,NN)
    return result

def sumsqr(x,w=None,axis=None,NN=True):
    """Calculates the sum of the squares of the values of an input array.

    Since this is not a normalized statistic use of non-integer weights should
    be carefully considered prior to use.
    
    Parameters:
        x : ndarray or ArrayOfErr
            The data set.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the sum of the squares is to be taken.  If none
            is given then the sum of the square will be taken over the entire 
            array.
    Returns:
        result : float
            The sum of squares of x.
    """
    x,w = weighttest(x,w)
    if NN:
        result = np.sum(w*x**2,axis=axis)
    else:
        result = np.nansum(w*x**2,axis=axis)
    return result

def stddev(x,w=None,axis=None,NN=True):
    """Calculates the sample standard deviation of an input array.

    Since this is not a normalized statistic use of non-integer weights should
    be carefully considered prior to use.
    
    Parameters:
        x : ndarray or ArrayOfErr
            The data set.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the standard deviation is to be taken.  If none
            is given then the standard deviation will be taken over the entire array.
    Returns:
        result : float
            The sample standard deviation of x.
            
    @todo: - check if the method works along predefined axis
    """
    x,w = weighttest(x,w)
    if NN:
        result = np.sqrt(sumsqrdev(x,w,axis,NN)/(np.sum(w)-1))
    else:
        result = np.sqrt(sumsqrdev(x,w,axis,NN)/(np.nansum(w)-1))
    return result

def stddevpop(x,w=None,axis=None,NN=True):
    """Calculates the population standard deviation of an input array.
    
    This is essentially a normalized version of sumsqrdev

    Parameters:
        x : ndarray or ArrayOfErr
            The data set.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the population standard deviation is to be 
            taken.  If none is given then the population standard deviation
            will be taken over the entire array.
    Returns:
        result : float
            The population standard deviation of x.
    See Also:
        sumsqrdev
    """
    x,w = weighttest(x,w)
    if NN:
        result = np.sqrt(sumsqrdev(x,w,axis,NN)/np.sum(w,axis=axis))
    else:
        result = np.sqrt(sumsqrdev(x,w,axis,NN)/np.nansum(w,axis=axis))
    return result

def absdev(x,w=None,axis=None,NN=True):
    """Calculates the average of the absolute deviaitions of an input array.

    Parameters:
        x : ndarray or ArrayOfErr
            The data set.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the average of the absolute deviations is to be
            taken.  If none is given then the averate of the absolute 
            deviations will be taken over the entire array.
    Returns:
        result : float
            The average absolute deviation of x.
    """
    x,w = weighttest(x,w)
    if NN:
        result = np.sum(w*np.abs(x - mean(x,w,axis,NN)),axis=axis)/np.sum(w,axis=axis)
    else:
        result = np.nansum(w*np.abs(x - mean(x,w,axis,NN)),axis=axis)/np.nansum(w,axis=axis)
    return result

def var(x,w=None,axis=None,NN=True):
    """Calculates sample variance of an input array.

    Parameters:
        x : ndarray or ArrayOfErr
            The data set.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the variance is to be taken.  If none is given
            then the variance will be taken over the entire array.
    Returns:
        result : float
            The sample variance of x.
    """
    result = stddev(x,w,axis,NN)**2
    return result

def varpop(x,w=None,axis=None,NN=True):
    """Calculates population variance of an input array.

    Parameters:
        x : ndarray or ArrayOfErr
            The data set.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the population variance is to be taken.  If 
            none is given then the population variance will be taken over 
            the entire array.
    Returns:
        result : float
            The population variance of x.
    """
    result = stddevpop(x,w,axis,NN)**2
    return result

def coefvar(x,w=None,axis=None,NN=True):
    """Calculates the sample coefficient of variance.

    Parameters:
        x : ndarray or ArrayOfErr
            The data set.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the coefficient of variance is to be taken.  If
            none is given then the coefficient of variance will be taken over 
            the entire array.
    Returns:
        result : float
            The sample coefficient of variance of x.
    """
    result = stddev(x,w,axis,NN)*100/mean(x,w,axis,NN)
    return result

def coefvarpop(x,w=None,axis=None,NN=True):
    """Calculates the population coefficient of variance.

    Parameters:
        x : ndarray or ArrayOfErr
            The data set.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the population coefficient of variance is to be
            taken.  If none is given then the population coefficient of 
            variance will be taken over the entire array.
    Returns:
        result : float
            The population coefficient of variance of x.
    """
    result = stddevpop(x,w,axis,NN)*100/mean(x,w,axis,NN)
    return result

def skewness(x,w=None,axis=None,NN=True):
    """Calculates the skewness of the input array.
    
    Skewness is a measure of symmetry, or more precisely, the lack of symmetry.
    A distribution, or data set, is symmetric if it looks the same to the left
    and right of the center point.  The skewness for a normal distribution is 
    zero, and any symmetric data should have a skewness near zero.  Negative 
    values for the skewness indicate data that are skewed left and positive 
    values for the skewness indicate data that are skewed right. By skewed 
    left, we mean that the left tail is heavier than the right tail. Similarly,
    skewed right means that the right tail is heavier than the left tail.
    
    Parameters:
        x : ndarray or ArrayOfErr
            The data set.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the skewness is to be taken.  If none is given
            then the skewness will be taken over the entire array.
    Returns:
        result : float
            The skewness of x.  Negative values indicate data that is skewed
            left.  Positive values indicate data that is skewed right.
    """
    u = moment(x,3,w,axis,NN)
    if NN:
        l = np.sum(w,axis=axis)*stddev(x,w,axis,NN)**3
    else:
        l = np.nansum(w,axis=axis)*stddev(x,w,axis,NN)**3
    result = u/l
    return result

def coefskewness(x,w=None,axis=None,NN=True):
    """Calculates the coefficient of skewness.
    
    Half of the skewness.
    
    Parameters:
        x : ndarray or ArrayOfErr
            The data set.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the coefficient of skewness is to be taken.  If
            none is given then the coefficient of skewness will be taken over 
            the entire array.
    Returns:
        result : float
            The coefficient of skewness of x.  Negative values indicate data 
            that is skewed left.  Positive values indicate data that is skewed 
            right.
    See Also:
        skewness
    """
    result = skewness(x,w,axis,NN)/2
    return result

def kurtosis(x,w=None,axis=None,NN=True):
    """Calculates the kurtosis of the input array.
    
    Kurtosis is a measure of whether the data are peaked or flat relative to a 
    normal distribution. That is, data sets with positive kurtosis tend to have
    a distinct peak near the mean, decline rather rapidly, and have heavy 
    tails. Data sets with negative kurtosis tend to have a flat top near the 
    mean rather than a sharp peak. A uniform distribution being the extreme 
    case.
    
    Parameters:
        x : ndarray or ArrayOfErr
            The data set.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the kurtosis is to be taken.  If none is given 
            then the kurtosis will be taken over the entire array.
    Returns:
        result : float
            The kurtosis of x.  Positive kurtosis indicates a sharp peak.
            Negative kurtosis indicates a weak peak.
    """
    u = moment(x,4,w,axis,NN)
    if NN:
        l = np.sum(w,axis=axis)*stddev(x,w,axis,NN)**4
    else:
        l = np.nansum(w,axis=axis)*stddev(x,w,axis,NN)**4
    result = u/l - 3
    return result

def coefkurtosis(x,w=None,axis=None,NN=True):
    """Calculates the coefficient of kurtosis.
    
    The kurtosis + 3.

    Parameters:
        x : ndarray or ArrayOfErr
            The data set.
        w : ndarray
            Optional.  The weights corresponding to each data point.
        NN : boolean
            If True (default) nan values in x will not be ignored and so nan
            will be returned if they are present.  If False, then nan values
            will be ignored in x and weights of nan will be treated as a weight
            of 0.
        axis : integer
            The axis over which the coefficient of kurtosis is to be taken.  If
            none is given then the coefficient of kurtosis will be taken over 
            the entire array.
    Returns:
        result : float
            The coefficient of kurtosis of x.
    See Also:
        kurtosis
    """
    result = kurtosis(x,w,axis,NN) + 3
    return result



#Private Functions    
    
def _chk_asarray(a, axis, dtype):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a, dtype)
        outaxis = axis
    return a, outaxis
    

    
#   
#    
#    x = np.random.randn(10)
#    mu = np.random.randn(10)
#    sigma = np.random.randn(10)
#    df = np.random.randint(1, 100, 10)
#    sigma *= sigma
#    
#    print np.log(tpdf(x, mu[2], sigma[2], df[2]))
#    print tlnpdf(x, mu[2], sigma[2], df[2])
#    
#    print np.log(tpdf(x, mu, sigma, df))
#    print tlnpdf(x, mu, sigma, df)
#    
#    print np.log(tpdf(x[2], mu, sigma, df))
#    
#    print np.log(normpdf(x, mu[2], sigma[2]))
#    print normlnpdf(x, mu[2], sigma[2])
#    
#    print 'gradient test:'
#    def _likel(params):
#        return tpdf(-2, params[0], params[1], params[2])
#    def _gradient(params):
#        return gradient_tpdf(-2, params[0], params[1], params[2])
#    params = np.array([mu[2], sigma[2], df[2]])
#    print 'likel={0}'.format(_likel(params))
#    print 'grad={0}'.format(_gradient(params))
#    print 'approx_grad={0}'.format(spopt.approx_fprime(params, _likel, np.sqrt(np.finfo(float).eps)))
#
#
#    print 'mv gradient test:'
#    def _likel(params):
#        L = params[3:12].reshape(3,3)
#        L = np.tril(L)
#        S = np.dot(L, L.T)
#        return mvtpdf(np.array([0.8, 1.2, -0.2]), params[0:3], S, params[12])
#    def _gradient(params):
#        return gradient_mvtpdf(np.array([0.8, 1.2, -0.2]), params[0:3], params[3:12].reshape(3,3), params[12])
#    params = np.array([0.4,0.3,-0.8, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 12])
#    print params[3:12].reshape(3,3)
#    print 'likel={0}'.format(_likel(params))
#    print 'grad={0}'.format(_gradient(params))
#    print 'approx_grad={0}'.format(spopt.approx_fprime(params, _likel, np.sqrt(np.finfo(float).eps)))
#        
#    
#    
#    x = np.random.randn(10,2)
#    mu = np.random.randn(10,2)
#    sigma = np.identity(2)
#    
#    print 'puff'
#    print mvnpdf(x, mu, sigma)
#    print np.exp(mvnlnpdf(x, mu, sigma))
#    
#    print 'hadsch'
#    print mvtpdf(x, mu, sigma, 3)
#    
#    sigma = np.tile(sigma, (3,1,1))
#    
#    
#    print normpdf(2, 0.6, 0.7)
#    print np.exp(normlnpdf(2, 0.6, 0.7))
#    
#    print tpdf(2,0,1,4)
#    print np.exp(tlnpdf(2,0,1,4))