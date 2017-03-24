'''
Created on Jul 9, 2012

@author: marcel
'''
import numpy as np 

from abc import ABCMeta, abstractmethod

class MeanFunction(object):
    
    __metaclass__ = ABCMeta
    
    __slots__ = ('_params',   #an array of the kernel hyperparameters
                 '_n',        #the number of hyperparameters
                 )
    
    def __init__(self, params):
        '''
        '''
        params = np.ravel(np.atleast_1d(np.asarray(params, dtype=np.float64)))
        
        self._params = params
        self._n = len(params)
        
    def __mul__(self, rhs):
        '''
        Composite two kernels by the product kernel.
        
        @return: The product of self and rhs. 
        '''
        return ProductMean(self, rhs)
    
    def __add__(self, rhs):
        '''
        Composite two kernels by the sum kernel.
        
        @return: The sum of self and rhs. 
        '''
        return SumMean(self, rhs)
        
    def __str__(self):
        """
        String representation of mean function
        """
        raise RuntimeError( """%s.__str__() Should have been implemented """
            """in base class""" % str(self.__class__) )
        
    @abstractmethod
    def __call__(self, X):
        """
        Returns value of kernel for the specified data points
        """
        pass
    
    @abstractmethod
    def derivate(self, i):
        '''
        Returns the derivative of the kernel with respect to the i'th parameter.
        '''
        self._range_check(i)
    
    def get_parameter(self, i):
        self._range_check(i)
        return self._params[i]
    
    def set_parameter(self, i, value):
        self._range_check(i)
        self._params[i] = value
    
    def _get_params(self):
        '''
        Return the hyperparameters of the kernel. Note, the returned array is 
        not immutable and a reference to the original, so that specific paramaters
        could be changed by the returned reference. 
        
        @return:  an array of hyperparameters 
        '''
        return self._params
    
    def _set_params(self, params):
        '''
        Sets the specified hyperparameters of the kernel. The method raise an 
        exception if the size of given parameter array not the same with the
        number of parameters needed by the kernel.
        '''
        params = np.ravel(np.array(params))
        if len(params) != self.nparams:
            raise TypeError('''wrong number of parameters.
                            {0} parameters are needed.'''.format(self.nparams))
            
        for i in xrange(len(params)):
            self._params[i] = params[i]
        
    params = property(fget=_get_params, fset=_set_params)
    
    def _number_of_params(self):
        return self._n
    
    nparams = property(fget=_number_of_params)
        
    def _range_check(self, index):
        '''
        '''
        if index < 0 or index > self._n-1:
            raise IndexError("Index %s out of bound." % index)
        
    def copy(self):
        params = np.copy(self._params)
        new_mfct = self.__class__(params)
        return new_mfct
    
    class _IDerivativeFun(object):

        __metaclass__ = ABCMeta

        @abstractmethod
        def __call__(self, X):
            return None
        
class CompositeMean(MeanFunction):
    '''
    '''
    
    __slots__ = ('_lhs',    #left hand side mean function of the compositum
                 '_rhs'     #right hand side mean function of the compositum
                 )
    
    def __init__(self, lhs, rhs):
        params = CompositeMean.MixedArray(lhs.params, rhs.params)
        #Kernel.__init__(self, params)
        
        self._params = params
        self._n = len(params)
        
        self._lhs = lhs
        self._rhs = rhs
        
    def _get_lhs(self):
        '''
        Return the kernel on the left hand side of the compositum.
        '''
        return self._lhs
        
    lhs = property(fget=_get_lhs)
    
    def _get_rhs(self):
        '''
        Return the kernel on the right hand side of the compositum.
        '''
        return self._rhs
        
    rhs = property(fget=_get_rhs)
    
    def meanfct_by_parameter(self, i):
        self._range_check(i)
        return self.lhs if i < self.lhs.n_params else self.rhs
    
    def _lookup_meanfct_and_param(self, i): 
        '''
        Returns a triplet of both kernels and the parameter index of the active 
        kernel as tuple. The first element of the tuple is the active kernel for
        which the parameter request is done. The second kernel is the passive 
        part. The returned tuple has the following form:
        
        (active kernel, passive kernel, param) 
        '''
        self._range_check(i)
        if i < self.lhs.nparams:
            return (self.lhs, self.rhs, i)
        else:
            return (self.rhs, self.lhs, i-self.lhs.nparams)
        
    def copy(self):
        new_lhs = self._lhs.copy()
        new_rhs = self._rhs.copy()
        new_mfct = self.__class__(new_lhs, new_rhs)
        return new_mfct
        
    class MixedArray(object):
        def __init__(self, a, b):
            self.a = a
            self.b = b
        
        def __len__(self):
            return len(self.a) + len(self.b)
        
        def __getitem__(self, i):
            array, idx = self.__array_idx_at(i)
            return array[idx]
        
        def __setitem__(self, i, value):
            array, idx = self.__array_idx_at(i)
            array[idx] = value

        def __array_idx_at(self, i):
            return (self.a, i) if i < len(self.a) else (self.b, i-len(self.a))
        
        def __str__(self):
            return str(np.r_[self.a, self.b]) 
        
class SumMean(CompositeMean):
    
    def __init__(self, lhs, rhs):
        CompositeMean.__init__(self, lhs, rhs)
        
    def __call__(self, X):
        return self.lhs(X) + self.rhs(X)

    def __str__( self ):
        return "SumMean({0},{1})".format(str(self.lhs), str(self.rhs))

    def derivate(self, i):
        '''
        Returns the derivative of the mean fct with respect to the i'th parameter.
        '''
        u, v, i = self._lookup_meanfct_and_param(i)
        return u.derivate(i)
    
class ProductMean(CompositeMean):
    
    def __init__(self, lhs, rhs):
        CompositeMean.__init__(self, lhs, rhs)
        
    def __call__(self, X):
        return self.lhs(X) * self.rhs(X) #elementwise multiplication
    
    def __str__( self ):
        return "ProductMean({0},{1})".format(str(self.lhs), str(self.rhs))

    def derivate(self, i):
        '''
        Returns the derivative of the mean fct with respect to the i'th parameter.
        '''
        u, v, i = self._lookup_meanfct_and_param(i)
        u_deriv = u.derivate(i)
        
        fun = lambda X: v(X) * u_deriv(X)
        return fun
    
class HiddenMean(MeanFunction):
    '''
    A wrapper mean function that hides (i.e. fixes) the parameters of another mean function
    
    To users of this class it appears as the kernel has no parameters to optimise.
    This can be useful when you have a mixture kernel and you only want to learn 
    one of sub kernel's parameters.
    
    @todo: - we have no read access to the parameters of the hidden mean fct (fix this)
    '''
    
    __slots__ = ('_hidden_mfct')
    
    def __init__(self, hidden_mfct):
        MeanFunction.__init__(self, np.asarray([]))
        self._hidden_mfct = hidden_mfct
        
    def __str__(self):
        return "HiddenMean"

    def __call__(self, X):
        return self._hidden_mfct(X)
    
    def derivate(self, i):
        pass
    
    def copy(self):
        cp_hidden_mfct = self._hidden_mfct.copy()
        return self.__class__(cp_hidden_mfct)
    
class MaskedFeatureMean(MeanFunction):
    '''
    '''
    
    __slots__ = ('_mfct',
                 '_mask'     #mask of the used features
                 )
    def __init__(self, mfct, mask):
        MeanFunction.__init__(self, np.array([]))
        
        self._params = mfct.params
        self._n = len(mfct.params)
        
        self._mfct = mfct
        self._mask = mask
    
    def __str__(self):
        return "MaskedFeatureMean"

    def __call__(self, X):
        mask = self._mask
        return self._mfct(X[:,mask])
    
    def derivate(self, i): 

        class _DerivativeFun(MeanFunction._IDerivativeFun):
            def __init__(self, mfct, mask, i):
                self.mfct = mfct
                self.mask = mask
                self.i = i
                
            def __call__(self, X):
                
                deriv = self.mfct.derivate(self.i)
                mask = self.mask
                return deriv(X[:,mask])
            
        fun = _DerivativeFun(self._kernel, self._mask, i)
        return fun
    
    def copy(self):
        '''
        @todo: .implement copy constructor
        '''
        pass
    
class FixedParameterMean(MeanFunction):
    '''
    @todo: - does not work correctly, because its parameters vector is just a copy 
             of the wrapped one. fix it!!!
    '''
    
    __slots__ = ('_mfct',       #mean funtion with fixed parameters 
                 '_param_map'   #mapping to the flexible parameters of the kernel
                 ) 
    
    def __init__(self, mfct, fixed_params):
        '''
        '''
        fixed_params = np.ravel(np.asarray(fixed_params))
        
        params = mfct.params
        n = mfct.n_params
        
        mask = np.empty(n, dtype=np.bool)
        mask[fixed_params] = False
        
        param_map = np.where(mask)
        
        MeanFunction.__init__(self, params[param_map])
        self._mfct = mfct
        self._param_map = param_map
    
    def __str__(self):
        return "FixedParameterMean"
    
    def __call__(self, X):
        return self._mfct(X)
    
    def derivate(self, i): 
        return self._meanfct.derivate(self._param_map[i])
    
    def copy(self):
        '''
        @todo: .implement copy constructor
        '''
        pass

class ConstantMean(MeanFunction):
    
    def __init__(self, c):
        MeanFunction.__init__(self, c)
        
    def __str__( self ):
        return "ConstantMean({0})".format(self.params[0])
    
    def __call__(self, X):
        c = self.params[0]
        M = np.ones(len(X))*c
        return M
        
    def derivate(self, i):
        class _DerivativeFun(MeanFunction._IDerivativeFun):
            def __init__(self, mfct, i):
                self.mfct = mfct
                self.i = i
                
            def __call__(self, X):
                dM = np.ones(len(X))
                return dM
            
        fun = _DerivativeFun(self, i)
        return fun
        
class LinearMean(MeanFunction):
    
    def __init__(self, w):
        MeanFunction.__init__(self, w)
        
    def __str__( self ):
        return "LinearMean({0})".format(self.params)
    
    def __call__(self, X):
        w = self.params
        M = np.dot(X,w)
        return M
        
    def derivate(self, i):
        class _DerivativeFun(MeanFunction._IDerivativeFun):
            def __init__(self, mfct, i):
                self.mfct = mfct
                self.i = i
                
            def __call__(self, X):
                i = self.i
                dM = X[:,i]
                return dM
            
        fun = _DerivativeFun(self, i)
        return fun
    
class BiasedLinearMean(MeanFunction):
    
    def __init__(self, w, bias):
        w = np.asarray(w).ravel()
        params = np.r_[w,bias]
        MeanFunction.__init__(self, params)
        
        self._d = len(w)
        
    def __str__( self ):
        return "BiasedLinearMean({0},{1})".format(self.params[0:self._d],self.params[self._d])
    
    def __call__(self, X):
        d = self._d
        w = self.params[0:d]
        bias = self.params[d]
    
        M = np.dot(X,w) + bias
        return M
        
    def derivate(self, i):
        class _DerivativeFun(MeanFunction._IDerivativeFun):
            def __init__(self, mfct, i):
                self.mfct = mfct
                self.i = i
                
            def __call__(self, X):
                d = self.mfct._d
                i = self.i
                
                if i < d:
                    dM = X[:,i]
                elif i == d:
                    dM = np.ones(len(X))
                else:
                    raise TypeError('Unknown hyperparameter')
                
                return dM
            
        fun = _DerivativeFun(self, i)
        return fun

