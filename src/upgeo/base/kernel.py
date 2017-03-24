'''
Created on Aug 2, 2011

@author: marcel

@todo: - remove precomputed distance matrix R
	   - implement deepth copy method
		
'''

import numpy as np
import scipy.optimize as spopt

from abc import ABCMeta, abstractmethod
from upgeo.util.metric import distance_matrix

class Kernel(object):
	'''
	Abstract base class for all gaussian process kernels. The hyperparmater are 
	defined in log space so that all implemented subclasses must consider this.
	
	@todo: - allow the definition of priors on hyperparameters
		   - naming hyperparameters (access by name)
		   - which dtype should be the hyperparameter array?
		   - validate the gradient wrt X of a diagonal cov matrix for the scalar kernels
	'''
	
	__metaclass__ = ABCMeta
	
	__slots__ = ('_params',   #an array of the kernel hyperparameters
				 '_n',		#the number of hyperparameters
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
		return ProductKernel(self, rhs)
	
	def __add__(self, rhs):
		'''
		Composite two kernels by the sum kernel.
		
		@return: The sum of self and rhs. 
		'''
		return SumKernel(self, rhs)
	
	def __str__(self):
		"""
		String representation of kernel
		"""
		raise RuntimeError( """%s.__str__() Should have been implemented """
			"""in base class""" % str(self.__class__) )
		
	@abstractmethod
	def __call__(self, X, Z=None, diag=False):
		"""
		Returns value of kernel for the specified data points
		"""
		pass
	
	#@todo: make it abstract
	def gradient(self, covGrad, X, Z=None, diag=False):
		pass
	
	def gradientX(self, covGrad, X, Z=None, diag=False):
		m,d = X.shape
		derivX = self.derivateX()
		dKx = derivX(X,Z,diag)
		gradX = np.zeros((m,d))
		
		for i in xrange(m):
			for j in xrange(d):
				gradX[i,j] = np.dot(dKx[:,i,j], covGrad[:,i])
				
		return gradX.flatten()
	
	@abstractmethod
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		'''
		self._range_check(i)
		
	@abstractmethod
	def derivateX(self):
		'''
		'''
		pass

	
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
		if len(params) != self.n_params:
			raise TypeError('''wrong number of parameters.
							{0} parameters are needed.'''.format(self.n_params))
			
		for i in xrange(len(params)):
			self._params[i] = params[i]
		
	params = property(fget=_get_params, fset=_set_params)
	
	def _number_of_params(self):
		return self._n
	
	n_params = property(fget=_number_of_params) #todo: remove
	
	nparams = property(fget=_number_of_params)
		
	def _range_check(self, index):
		'''
		'''
		if index < 0 or index > self._n-1:
			raise IndexError("Index %s out of bound." % index)
		
	def copy(self):
		params = np.copy(self._params)
		new_kernel = self.__class__(params)
		return new_kernel
	
	class _IDerivativeFun(object):

		__metaclass__ = ABCMeta

		@abstractmethod
		def __call__(self, X, Z=None, diag=False):
			return None

	class _IDerivativeFunX(object):

		__metaclass__ = ABCMeta

		@abstractmethod
		def __call__(self, x, Z):
			return None

class ZeroKernel(Kernel):
	def __init__(self):
		'''
		Initialize the constant kernel.
		
		@arg const: the constant parameter.
		'''
		Kernel.__init__(self, [])
		
	def __call__(self, X, Z=None, diag=False):
		'''
		'''
		
		xeqz = (Z == None)
		m = np.size(X, 0)
		if xeqz:
			if diag:
				K = np.zeros(m)
			else:
				K = np.zeros((m,m))
		else:
			n = np.size(Z, 0)
			K = np.zeros((m,n))
		
		return K
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		pass
	 
	def derivate(self, i):
		pass
	
	def derivateX(self, i):
		pass

	def copy(self):
		return self.__class__()
	
class ConstantKernel(Kernel):
	'''
	Covariance Kernel for a constant function. The covariance kernel is parametrized
	as:
	
	k(x_p,x_q) = c
	
	'''
	def __init__(self, const=0.0):
		'''
		Initialize the constant kernel.
		
		@arg const: the constant parameter.
		'''
		Kernel.__init__(self, const)
	
	def __call__(self, X, Z=None, diag=False):
		'''
		'''
		c = np.exp(self.params[0])
		
		xeqz = (Z == None)
		m = np.size(X, 0)
		if xeqz:
			if diag:
				K = np.ones(m)
			else:
				K = np.ones((m,m))
		else:
			n = np.size(Z, 0)
			K = np.ones((m,n))
		
		K *= c
		return K
	
	def __str__( self ):
		return "ConstantKernel({0})".format(self.params[0])
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		c = np.exp(self.kernel.params[0])
		xeqz = (Z == None)
		m = np.size(X, 0)
		if xeqz:
			if diag:
				dK = np.ones(m)
			else:
				dK = np.ones((m,m))
		else:
			n = np.size(Z, 0)
			dK = np.ones((m,n))
			
		dK *= c
		grad = np.array([np.sum(covGrad*dK)])
		return grad

	
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				c = np.exp(self.kernel.params[0])
				
				xeqz = (Z == None)
				m = np.size(X, 0)
				
				if xeqz:
					if diag:
						dK = np.ones(m)
					else:
						dK = np.ones((m,m))
				else:
					n = np.size(Z, 0)
					dK = np.ones((m,n))
				
				dK *= c
				return dK
			
		fun = _DerivativeFun(self)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					dK = np.zeros((m,n,d))
				return dK
				
		return _DerivativeFun(self)

	
class SqConstantKernel(Kernel):
	'''
	Covariance Kernel for a squared constant function. The covariance kernel is parametrized
	as:
	
	k(x_p,x_q) = c**2
	
	'''
	def __init__(self, const=0.0):
		'''
		'''
		Kernel.__init__(self, const)
		
	def __call__(self, X, Z=None, diag=False):
		'''
		'''
		c = np.exp(2*self.params[0])
		
		xeqz = (Z == None)
		m = np.size(X, 0)
		
		if xeqz:
			if diag:
				K = np.ones(m)
			else:
				K = np.ones((m,m))
		else:
			n = np.size(Z, 0)
			K = np.ones((m,n))
		
		K *= c
		return K
	
	def __str__( self ):
		'''
		'''
		return "SqConstantKernel({0})".format(self.params[0]**2.0)

	def gradient(self, covGrad, X, Z=None, diag=False):
		c = np.exp(2.0*self.params[0])
		xeqz = (Z == None)
		m = np.size(X, 0)
		if xeqz:
			if diag:
				dK = np.ones(m)
			else:
				dK = np.ones((m,m))
		else:
			n = np.size(Z, 0)
			dK = np.ones((m,n))
			
		dK *= 2.0*c
		grad = np.array([np.sum(covGrad*dK)])
		return grad
		
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				c = np.exp(2.0*self.kernel.params[0])
				
				xeqz = (Z == None)
				m = np.size(X, 0)
				
				if xeqz:
					if diag:
						dK = np.ones(m)
					else:
						dK = np.ones((m,m))
				else:
					n = np.size(Z, 0)
					dK = np.ones((m,n))
			
				dK *= 2.0*c
				return dK
			
		fun = _DerivativeFun(self)
		return fun
		
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					dK = np.zeros((m,n,d))
				return dK
				
		return _DerivativeFun(self)
			 
	
		
class CompositeKernel(Kernel):
	'''
	'''
	
	__slots__ = ('_lhs',	#left hand side kernel of the compositum
				 '_rhs'	 #right hand side kernel of the compositum
				 )
	
	def __init__(self, lhs, rhs):
		params = CompositeKernel.MixedArray(lhs.params, rhs.params)
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
	
	def kernel_by_parameter(self, i):
		self._range_check(i)
		return self.lhs if i < self.lhs.n_params else self.rhs
	
	def _lookup_kernel_and_param(self, i): 
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
		new_kernel = self.__class__(new_lhs, new_rhs)
		return new_kernel
		
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
		
class ProductKernel(CompositeKernel):
	
	def __init__(self, lhs, rhs):
		CompositeKernel.__init__(self, lhs, rhs)
		
	def __call__(self, X, Z=None, diag=False):
		return self.lhs(X,Z,diag=diag) * self.rhs(X,Z,diag=diag) #elementwise multiplication
	
	def __str__( self ):
		return "ProductKernel({0},{1})".format(str(self.lhs), str(self.rhs))

	def gradient(self, covGrad, X, Z=None, diag=False):
		lhs = self._lhs
		rhs = self._rhs
		
		Klhs = lhs(X,Z,diag)
		Krhs = rhs(X,Z,diag)
		
		#@todo: check if the gradient works correctly
		grad = np.empty(self.nparams)
		grad[:lhs.nparams] = lhs.gradient(covGrad*Krhs, X, Z, diag)
		grad[lhs.nparams:] = rhs.gradient(covGrad*Klhs, X, Z, diag)
		return grad


	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		'''
		u, v, i = self._lookup_kernel_and_param(i)
		u_deriv = u.derivate(i)
		
		fun = lambda X, Z=None, diag=False: v(X,Z,diag) * u_deriv(X,Z,diag)
		return fun

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				kernel = self.kernel
				lhs = kernel.lhs
				rhs = kernel.rhs
				dX_lhs = kernel.lhs.derivateX()
				dX_rhs = kernel.rhs.derivateX()
				
				dK_lhs = dX_lhs(X,Z,diag)
				dK_rhs = dX_rhs(X,Z,diag)
				K_lhs = lhs(X,Z,diag)
				K_rhs = rhs(X,Z,diag)
				
				d = X.shape[1]
				dK = np.empty(dK_lhs.shape)
				for i in xrange(d):
					dK[:,:,i] = dK_lhs[:,:,i]*K_rhs.T + dK_rhs[:,:,i]*K_lhs.T
				
				return dK
					
		return _DerivativeFun(self)
	

class SumKernel(CompositeKernel):
	
	def __init__(self, lhs, rhs):
		CompositeKernel.__init__(self, lhs, rhs)
		
	def __call__(self, X, Z=None, diag=False):
		return self.lhs(X,Z,diag=diag) + self.rhs(X,Z,diag=diag)

	def __str__( self ):
		return "SumKernel({0},{1})".format(str(self.lhs), str(self.rhs))

	def gradient(self, covGrad, X, Z=None, diag=False):
		lhs = self._lhs
		rhs = self._rhs
		
		#@todo: check if the gradient works correctly
		grad = np.empty(self.nparams)
		grad[:lhs.nparams] = lhs.gradient(covGrad, X, Z, diag)
		grad[lhs.nparams:] = rhs.gradient(covGrad, X, Z, diag)
		return grad

	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		u, v, i = self._lookup_kernel_and_param(i)
		return u.derivate(i)	

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				kernel = self.kernel
				dX_lhs = kernel.lhs.derivateX()
				dX_rhs = kernel.rhs.derivateX()
				
				dK = dX_lhs(X,Z,diag) + dX_rhs(X,Z,diag)
				return dK
					
		return _DerivativeFun(self)

class HiddenKernel(Kernel):
	'''
	A wrapper kernel that hides (i.e. fixes) the parameters of another kernel
	
	To users of this class it appears as the kernel has no parameters to optimise.
	This can be useful when you have a mixture kernel and you only want to learn 
	one child kernel's parameters.
	
	@todo: - we have no read access to the parameters of the hidden kernel (fix this)
	'''
	
	__slots__ = ('_hidden_kernel')
	
	def __init__(self, hidden_kernel):
		Kernel.__init__(self, np.asarray([]))
		self._hidden_kernel = hidden_kernel
		
	def __str__(self):
		return "HiddenKernel"

	def __call__(self, X, Z=None, diag=False):
		return self._hidden_kernel(X, Z, diag=diag)
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		return np.empty(0)
	
	def derivate(self, i):
		pass
	
	def derivateX(self):
		return self._hidden_kernel.derivateX()
	
	def copy(self):
		cp_hidden_kernel = self._hidden_kernel.copy()
		return self.__class__(cp_hidden_kernel)

class MaskedFeatureKernel(Kernel):
	'''
	todo : param problems
	'''
	
	__slots__ = ('_kernel',
				 '_mask' 	#mask of the used features
				 )
	def __init__(self, kernel, mask):
		Kernel.__init__(self, np.array([]))
		
		self._params = kernel.params
		self._n = len(kernel.params)
		
		self._kernel = kernel
		self._mask = mask
	
	def __str__(self):
		return "MaskedFeatureKernel"

	def __call__(self, X, Z=None, diag=False):
		mask = self._mask
		if Z == None:
			return self._kernel(X[:,mask],diag=diag)
		else:
			return self._kernel(X[:,mask],Z[:,mask],diag=diag)
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		''' 
		'''
		mask = self._mask
		if Z == None:
			return self._kernel.gradient(covGrad, X[:,mask], diag=diag)
		else:
			return self._kernel.gradient(covGrad, X[:,mask], Z[:,mask], diag=diag)

	
	def derivate(self, i): 

		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, mask, i):
				self.kernel = kernel
				self.mask = mask
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				
				deriv = self.kernel.derivate(self.i)
				mask = self.mask
				
				if Z == None:
					return deriv(X[:,mask], diag=diag)
				else:
					return deriv(X[:,mask], Z[:,mask], diag=diag)
				
				
			
		fun = _DerivativeFun(self._kernel, self._mask, i)
		return fun
	
	def derivateX(self):
		'''
		Does not work if we compute the diag of the gradient fix it!
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, mask):
				self.kernel = kernel
				self.mask = mask
				
			def __call__(self, X, Z=None, diag=False):
				deriv = self.kernel.derivateX()
				mask = self.mask
		
				xeqz = (Z==None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
				if diag:
					dKx = np.zeros((n*d))
				else:
					dKx = np.zeros((m,n,d))
					
				if Z == None:
					dKx[:,:, mask] = deriv(X[:,mask], diag=diag)
				else:
					dKx[:,:, mask] = deriv(X[:,mask], Z[:,mask], diag=diag)
				
				return dKx
		
		return _DerivativeFun(self._kernel, self._mask)
	
	def copy(self):
		'''
		@todo: .implement copy constructor
		'''
		cpyKernel = MaskedFeatureKernel(self._kernel.copy(), self._mask)
		return cpyKernel
	
	


class FixedParameterKernel(Kernel):
	'''
	@todo: - works only as high level order kernel if we composite different kernels 
	'''
	
	__slots__ = ('_kernel',	 #kernel with fixed parameters 
				 '_param_map'   #mapping to the flexible parameters of the kernel
				 ) 
	
	def __init__(self, kernel, fixed_params):
		'''
		'''
		fixed_params = np.ravel(np.asarray(fixed_params))
		
		params = np.asarray(kernel.params)
		n = kernel.n_params
		
		mask = np.ones(n, dtype=np.bool)
		mask[fixed_params] = False
		param_map = np.where(mask)[0]
		
		Kernel.__init__(self, params[param_map])
		self._kernel = kernel
		self._param_map = param_map
	
	def __str__(self):
		return "FixedParameterKernel"
	
	def __call__(self, X, Z=None, diag=False):
		return self._kernel(X,Z,diag=diag)
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		'''
		I think its faster to compute the gradient for all parameters
		and then extract the unmasked features.
		'''
		grad = self._kernel.gradient(covGrad, X, Z, diag)
		return grad[self._param_map]

	
	def derivate(self, i): 
		return self._kernel.derivate(self._param_map[i])
	
	def derivateX(self):
		return self._kernel.derivateX()
	
	def set_parameter(self, i, value):
		self._range_check(i)
		self._params[i] = value
		self._kernel._params[self._param_map[i]] = value

	
	def _set_params(self, params):
		'''
		Sets the specified hyperparameters of the kernel. The method raise an 
		exception if the size of given parameter array not the same with the
		number of parameters needed by the kernel.
		'''
		params = np.ravel(np.array(params))
		if len(params) != self.n_params:
			raise TypeError('''wrong number of parameters.
							{0} parameters are needed.'''.format(self.n_params))
			
		for i in xrange(len(params)):
			self._params[i] = params[i]
			self._kernel._params[self._param_map[i]] = params[i] 

	
	params = property(fget=Kernel._get_params, fset=_set_params)

	
	def copy(self):
		'''
		@todo: .implement copy constructor
		'''
		kernel = self._kernel
		nparams = kernel.nparams 
		
		mask = np.ones(nparams, dtype=np.bool)
		mask[self._param_map] = False
		fixed_params = np.where(mask)[0]
		
		cpyKernel = FixedParameterKernel(kernel.copy(), fixed_params)
		return cpyKernel

class NoiseKernel(Kernel):
	'''
	Independent covariance kernel, i.e. 'white noise', with specified variance.
	The covariance function is specified as:
	
	k(x_q, x_p) = s**2 * \delta(p,q)
	
	where s is the noise variance and \delta(p,q) is a Kronecker delta function
	which is 1 iff p==q and zero otherwise.
	'''
	
	def __init__(self, s=0.0):
		Kernel.__init__(self, np.asarray([s]))
	
	def __str__( self ):
		return "NoiseKernel({0})".format(self.params[0])
	
	def __call__(self, X, Z=None, diag=False):
		s = np.exp(2.0*self.params[0])
		
		xeqz = (Z == None)
		m = np.size(X, 0)
		
		if xeqz:
			if diag:
				K = np.ones(m)*s
			else:
				K = np.diag(np.ones(m)*s)
		else:
			n = np.size(Z, 0)
			K = np.zeros((m,n))
		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		s = np.exp(2.0*self.params[0])
				
		xeqz = (Z == None)
		m = np.size(X, 0)
				
		if xeqz:
			dK = 2.0*s*np.ones(m)
			if not diag:
				covGrad = np.diag(covGrad)
			grad = np.array([np.sum(covGrad*dK)])
		else:
			grad = np.array([0])			

		return grad

	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				s = np.exp(2.0*self.kernel.params[0])
				
				xeqz = (Z == None)
				m = np.size(X, 0)
				
				if xeqz:
					if diag:
						dK = 2.0*s*np.ones(m)
					else:
						dK = np.diag(2.0*s*np.ones(m)) 
				else:
					n = np.size(Z, 0)
					dK = np.zeros((m,n))
					
					#K *= 2.0#*params[0]
				return dK
			
		fun = _DerivativeFun(self)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					dK = np.zeros((m,n,d))
				return dK
				
		return _DerivativeFun(self)

class TaskNoiseKernel(Kernel):
	'''
	- Check gradient x
	'''
	__slots__ = ('_task_ids',
				 '_task_idx')
	
	def __init__(self, task_ids, task_idx, s=0.0):
		task_ids = np.unique(task_ids)
		n = len(task_ids)
		
		Kernel.__init__(self, np.ones(n)*s)
		self._task_ids = task_ids
		self._task_idx = task_idx
	
	def __str__( self ):
		return "TaskNoiseKernel({0})".format(self.params)
	
	def __call__(self, X, Z=None, diag=False):
		s = np.exp(2.0*self.params)
		
		xeqz = (Z == None)
		m = np.size(X, 0)
		
		if xeqz:
			task_ids = self._task_ids
			task_idx = self._task_idx
			n = self.nparams
			K = np.zeros(m)
			
			for i in xrange(n):
				K[X[:,task_idx] == task_ids[i]] = s[i]
				
			if not diag:
				K = np.diag(K)
				
		else:
			n = np.size(Z, 0)
			K = np.zeros((m,n))

		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		s = np.exp(2.0*self.params)
				
		xeqz = (Z == None)
		#m = np.size(X, 0)
			
		n = self.nparams
		grad = np.zeros(n)
		if xeqz:
			task_ids = self._task_ids
			task_idx = self._task_idx
			#dK = np.zeros(m)
			
			if not diag:
				covGrad = np.diag(covGrad)
			
			for i in xrange(n):
				dK = 2*s[i]
				grad[i] = np.array([np.sum(covGrad[X[:,task_idx] == task_ids[i]]*dK)])
					

		return grad
		
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				s = np.exp(2.0*self.kernel.params)
				i = self.i
				
				xeqz = (Z == None)
				m = np.size(X, 0)
				
				if xeqz:
					task_ids = self.kernel._task_ids
					task_idx = self.kernel._task_idx
					dK = np.zeros(m)
					dK[X[:,task_idx] == task_ids[i]] = 2.0*s[i]
					
					
					if not diag:
						dK = np.diag(dK) 
				else:
					n = np.size(Z, 0)
					dK = np.zeros((m,n))
					
					#K *= 2.0#*params[0]
				return dK
			
		fun = _DerivativeFun(self,i)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					dK = np.zeros((m,n,d))
				return dK
				
		return _DerivativeFun(self)


class GroupNoiseKernel(Kernel):
	'''
	Independent covariance kernel, i.e. 'white noise', with specified variance.
	The covariance function is specified as:
	
	k(x_q, x_p) = s**2 * \delta(p_i,q_i)
	
	where s is the noise variance and \delta(p,q) is a Kronecker delta function
	which is 1 iff p==q and zero otherwise.
	
	TODO: check efficient of kernel computation, maybe its iterate by itself or sorting the vectors before
	- Check gradient x
	'''
	
	__slots__ = ('_group_idx')
	
	def __init__(self, group_idx, s=0.0):
		Kernel.__init__(self, np.asarray([s]))
		self._group_idx = group_idx
	
	def __str__( self ):
		return "GroupNoiseKernel({0})".format(self.params[0])
	
	def __call__(self, X, Z=None, diag=False):
		s = np.exp(2.0*self.params[0])
		i = self._group_idx
		
		xeqz = (Z == None)
		if xeqz:
			if diag:
				#K = np.diag(np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int)) * s
				K = np.ones(len(X)) * s
			else:
				K = np.array((np.equal.outer(X[:,i],X[:,i])),dtype=float) * s
		else:
			K = np.array((np.equal.outer(X[:,i],Z[:,i])),dtype=float) * s
	
		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		s = np.exp(2.0*self.params[0])
		i = self._group_idx
		
		
		xeqz = (Z == None)
		if xeqz:
			if diag:
				#dK = np.diag(np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int)) * 2.0*s
				dK = np.ones(len(X)) * 2.0*s
			else:
				dK = np.array((np.equal.outer(X[:,i],X[:,i])),dtype=float) * 2.0*s
		else:
			dK = np.array((np.equal.outer(X[:,i],Z[:,i])),dtype=float) * 2.0*s
	
		grad = np.array([np.sum(covGrad*dK)])
		return grad


	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False, K=None):
				s = np.exp(2.0*self.kernel.params[0])
				i = self.kernel._group_idx
				
				
				xeqz = (Z == None)
				if xeqz:
					if diag:
						#dK = np.diag(np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int)) * 2.0*s
						dK = np.ones(len(X)) * 2.0*s
					else:
						dK = np.array((np.equal.outer(X[:,i],X[:,i])),dtype=float) * 2.0*s
				else:
					dK = np.array((np.equal.outer(X[:,i],Z[:,i])),dtype=float) * 2.0*s
			
				return dK
			
		fun = _DerivativeFun(self)
		return fun
	

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					dK = np.zeros((m,n,d))
				return dK
				
		return _DerivativeFun(self)
	
	def copy(self):
		cpyKernel = GroupNoiseKernel(self._group_idx, self._params[0])
		return cpyKernel

class TaskGroupNoiseKernel(Kernel):
	'''
	Independent covariance kernel, i.e. 'white noise', with specified variance.
	The covariance function is specified as:
	
	k(x_q, x_p) = s**2 * \delta(p_i,q_i)
	
	where s is the noise variance and \delta(p,q) is a Kronecker delta function
	which is 1 iff p==q and zero otherwise.
	
	TODO: check efficient of kernel computation, maybe its iterate by itself or sorting the vectors before
	- Check gradient x
	- implement copy constructor
	'''
	
	__slots__ = ('_group_idx',
				 '_task_idx',
				 '_task_ids')
	
	def __init__(self, task_ids, task_idx, group_idx, s=0.0):
		task_ids = np.unique(task_ids)
		n = len(task_ids)
		
		Kernel.__init__(self, np.ones(n)*s)
		
		self._group_idx = group_idx
		self._task_idx = task_idx
		self._task_ids = task_ids
	
	def __str__( self ):
		return "TaskGroupNoiseKernel({0})".format(self.params[0])
	
	def __call__(self, X, Z=None, diag=False):
		s = np.exp(2.0*self.params)
		i = self._group_idx
		j = self._task_idx
		task_ids = self._task_ids
		n = self.nparams
		xeqz = (Z == None)
		if xeqz:
			if diag:
			
				m = X.shape[0]
				B = np.ones(m)
			else:
				B = np.array(np.equal.outer(X[:,i],X[:,i])*np.equal.outer(X[:,j],X[:,j]),dtype=float)
		else:
			B = np.array(np.equal.outer(X[:,i],Z[:,i])*np.equal.outer(X[:,j],Z[:,j]),dtype=float)
		
		K = np.zeros(B.shape)
		for l in xrange(n):
			K[X[:,j] == task_ids[l]] = B[X[:,j] == task_ids[l]] * s[l]
	
		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		s = np.exp(2.0*self.params)
		i = self._group_idx
		j = self._task_idx
		task_ids = self._task_ids
		n = self.nparams
		xeqz = (Z == None)
		if xeqz:
			if diag:
			
				m = X.shape[0]
				B = np.ones(m)
			else:
				B = np.array(np.equal.outer(X[:,i],X[:,i])*np.equal.outer(X[:,j],X[:,j]),dtype=float)
		else:
			B = np.array(np.equal.outer(X[:,i],Z[:,i])*np.equal.outer(X[:,j],Z[:,j]),dtype=float)
		
		
		grad = np.zeros(n)
		for l in xrange(n):
			dK = 2.0*s[l]
			grad[l] =  np.sum(B[X[:,j] == task_ids[l]] * covGrad[X[:,j] == task_ids[l]] * dK) 
			
		return grad

	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, l):
				self.kernel = kernel
				self.l = l
				
			def __call__(self, X, Z=None, diag=False, K=None):
				s = np.exp(2.0*self.params)
				i = self.kernel._group_idx
				j = self.kernel._task_idx
				l = self.l
				task_ids = self.kernel._task_ids
				xeqz = (Z == None)
				if xeqz:
					if diag:
					
						m = X.shape[0]
						B = np.ones(m)
					else:
						B = np.array(np.equal.outer(X[:,i],X[:,i])*np.equal.outer(X[:,j],X[:,j]),dtype=float)
				else:
					B = np.array(np.equal.outer(X[:,i],Z[:,i])*np.equal.outer(X[:,j],Z[:,j]),dtype=float)
				
				dK = np.zeros(B.shape)
				dK[X[:,j] == task_ids[l]] = B[X[:,j] == task_ids[l]] * 2*s[l]
			
				return dK
						
		fun = _DerivativeFun(self,i)
		return fun
	

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					dK = np.zeros((m,n,d))
				return dK
				
		return _DerivativeFun(self)
	

class CorrelatedNoiseKernel(Kernel):
	'''
	Independent covariance kernel, i.e. 'white noise', with specified variance.
	The covariance function is specified as:
	
	k(x_q, x_p) = s**2 * \delta(p_i,q_i)
	
	where s is the noise variance and \delta(p,q) is a Kronecker delta function
	which is 1 iff p==q and zero otherwise.
	
	TODO: check efficient of kernel computation, maybe its iterate by itself or sorting the vectors before
	- Check gradient x
	'''
	
	__slots__ = ('_group_idx')
	
	def __init__(self, group_idx, r, s=0.0):
		Kernel.__init__(self, np.asarray([r, s]))
		self._group_idx = group_idx
	
	def __str__( self ):
		return "GroupNoiseKernel({0})".format(self.params[0])
	
	def __call__(self, X, Z=None, diag=False):
		#r = 1.0/(1.0+np.exp(self.params[0])) #constraining r to be in the range [0,1]
		r = np.exp(self.params[0])
		s = np.exp(2.0*self.params[1])
		i = self._group_idx
		
		xeqz = (Z == None)
		if xeqz:
			if diag:
				K = np.diag(np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int)) * s
			else:
				K = np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int) * s / (r+1.0)
				di = np.diag_indices(len(X))
				#K[di] /= r
				K[di] *= (r+1.0)
		else:
			K = np.array((np.equal.outer(X[:,i],Z[:,i])),dtype=int) * s / (r+1.0)
			
	
		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		r = np.exp(self.params[0])
		s = np.exp(2.0*self.params[1])
		i = self._group_idx
		
		
		xeqz = (Z == None)
		if xeqz:
			if diag:
				K = np.diag(np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int)) * s
			else:
				K = np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int) * s / (r+1.0)
				di = np.diag_indices(len(X))
				K[di] *= (r+1.0)
		else:
			K = np.array((np.equal.outer(X[:,i],Z[:,i])),dtype=int) * s / (r+1.0)
	
		grad = np.zeros(2)
		dK = -K*r/(r+1.0)
		if xeqz:
			if diag:
				dK = np.zeros(len(X))
			else:
				di = np.diag_indices(len(X))
				dK[di] = 0
		grad[0] = np.sum(covGrad*dK)
		grad[1] = 2.0*np.sum(covGrad*K)
		return grad

	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False, K=None):
				r = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
				grp_idx = self.kernel._group_idx
				i = self.i
				
				
				xeqz = (Z == None)
				if xeqz:
					if diag:
						K = np.diag(np.array((np.equal.outer(X[:,grp_idx],X[:,grp_idx])),dtype=int)) * s
						#dK = np.diag(np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int)) * 2.0*s
					else:
						#K = np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int) * 2.0*s
						K = np.array((np.equal.outer(X[:,grp_idx],X[:,grp_idx])),dtype=int) * s / (r+1.0)
						di = np.diag_indices(len(X))
						K[di] *= r+1.0
						#dK = np.array((np.equal.outer(X[:,i],X[:,i])),dtype=int) * 2.0*s
				else:
					K = np.array((np.equal.outer(X[:,grp_idx],Z[:,grp_idx])),dtype=int) * s / (r+1.0)
					
				if i == 0:
					dK = -K*r/(r+1.0)
					if xeqz:
						if diag:
							dK = np.zeros(len(X))
						else:
							di = np.diag_indices(len(X))
							dK[di] = 0
				elif i == 1:
					dK = 2.0*K
				else:
					raise ValueError('Unknown hyperparameter: {0}'.format(i))
		
				return dK
			
		fun = _DerivativeFun(self, i)
		return fun
	

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					dK = np.zeros((m,n,d))
				return dK
				
		return _DerivativeFun(self)

	def copy(self):
		cpyKernel = CorrelatedNoiseKernel(self._group_idx, self._params[0], self._params[1])
		return cpyKernel

class RBFKernel(Kernel):
	'''
	Exponential covariance kernel with isotropic distance measure. 
	The kernel is parametrized as:
	
	k(x_p, x_q) = s**2 * exp(-|x_p-x_q| / l) 
	
	'''
	
	def __init__(self, l, s=0.0):
		Kernel.__init__(self, np.asarray([l,s]))
		
	def __str__( self ):
		return "RbfKernel({0},{1})".format(self.params[0],self.params[1])
	
	def __call__(self, X, Z=None, diag=False):
		
		l = np.exp(self.params[0])
		s = np.exp(2.0*self.params[1])
		xeqz = (Z == None)
		
		if xeqz and diag:
			R = np.zeros(X.shape[0])
		else:
			R = distance_matrix(X, Z, metric='euclidean')
			
		K = s * np.exp(-R/l)
		return K
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		'''
		'''
		l = np.exp(self.params[0])
		s = np.exp(2.0*self.params[1])
		xeqz = (Z == None)
		
		grad = np.zeros(2)


		if xeqz and diag:
			R = np.zeros(X.shape[0])
		else:
			R = distance_matrix(X, Z, metric='euclidean')
		K = s * np.exp(-R/l)
		
		#gradient of the length scale l
		dKl = K * R/l
		grad[0] = np.sum(covGrad*dKl)
		
		#gradient of the signal variance
		dKs = 2.0 * K
		grad[1] = np.sum(covGrad*dKs)

		return grad

			
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				l = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
				xeqz = (Z == None)
				
				if xeqz and diag:
					R = np.zeros(X.shape[0])
				else:
					R = distance_matrix(X, Z, metric='euclidean')
			
				if self.i == 0:
					#gradient of scale parameter l
					dK = s * np.exp(-R/l) * R / l 
				elif self.i == 1:
					#gradient of the variance parameter s
					dK = 2.0*s * np.exp(-R/l)
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				return dK
			
		fun = _DerivativeFun(self, i)
		return fun

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				l = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
		
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					R = distance_matrix(X, Z, metric='euclidean')
					K = s * np.exp(-R/l)
					dK = np.zeros((m,n,d))
					R = R + 1e-16 #prevent division by zeros
					for i in xrange(d):
						dK[:,:,i] = np.add.outer(Z[:,i],-X[:,i]) * K.T / (R.T*l)
				return dK
				
		return _DerivativeFun(self)

	
	def copy(self):
		cp_kernel = RBFKernel(self._params[0], self._params[1])
		return cp_kernel

class ARDRBFKernel(Kernel):
	'''
	RBF covariance kernel with automatic relevance determinition
	distance measure. The kernel is parametrized as:
	
	k(x_p, x_q) = s**2 * exp(-(x_p-x_q)' * inv(L) *  (x_p-x_q)/ 2) 
	'''
	
	__slots__ = ('_d')
	
	def __init__(self, l, s=0.0):
		l = np.asarray(l).ravel()
		params = np.r_[l,s]
		Kernel.__init__(self, params)
		
		self._d = len(l)
		
	def __str__( self ):
		return "ARDSEKernel()"
	
	def __call__(self, X, Z=None, diag=False):
		'''
		@todo: - optimize the distance computation of the diagonal dot prod
		'''
		
		xeqz = (Z == None)
			
		d = self._d
		l = np.exp(self.params[0:d])
		s = np.exp(2.0*self.params[d])
			
		X = X/l
		if xeqz:	
			if diag:
				K = np.zeros(X.shape[0])
			else:
				K = distance_matrix(X,  metric='euclidean') 
		else:
			Z = Z/l
			K = distance_matrix(X, Z, metric='euclidean')
			
		
		K = s*np.exp(-K)
		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		'''
		'''
		xeqz = (Z == None)
					
		d = self.kernel._d
		l = np.exp(self.kernel.params[0:d])
		s = np.exp(2.0*self.kernel.params[d])

		grad = np.zeros(d+1)

		X = X/l
		if xeqz:
			if diag:
				R = np.zeros(X.shape[0])
			else:
				R = distance_matrix(X, metric='euclidean') 
		else:
			Z = Z/l
			R = distance_matrix(X, Z, metric='euclidean')
		K = s*np.exp(-R)
		
		#gradient of the length scales l
		R  = R+1e-16
		for i in xrange(d):
			if xeqz:
				if diag:
					Kprime = 0.0
				else:
					Kprime = distance_matrix(X[:,i,np.newaxis], metric='sqeuclidean')
			else:
				Kprime = distance_matrix(X[:,i,np.newaxis], 
										 Z[:,i,np.newaxis], metric='sqeuclidean')
						
			dKl = K*Kprime/R
			grad[i] = np.sum(covGrad*dKl)
		
		#gradient of the signal variance
		dKs = 2.0 * K
		grad[d+1] = np.sum(covGrad*dKs)

		return grad

			
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - check the gradient of the length scale vector (gradient check failed, 
				 but its identical to matlab gpml implementation)
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):	 
				xeqz = (Z == None)
					
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
					
				X = X/l
				if xeqz:
					if diag:
						R = np.zeros(X.shape[0])
					else:
						R = distance_matrix(X, metric='euclidean') 
				else:
					Z = Z/l
					R = distance_matrix(X, Z, metric='euclidean')
				
				K = s*np.exp(-R)
				if self.i < d:
					if xeqz:
						if diag:
							Kprime = 0.0
						else:
							Kprime = distance_matrix(X[:,i,np.newaxis], metric='sqeuclidean')
					else:
						Kprime = distance_matrix(X[:,i,np.newaxis], 
												 Z[:,i,np.newaxis], metric='sqeuclidean')
						
					R  = R+1e-16
					dK = K*Kprime/R
				elif self.i == d:
					#gradient of the variance parameter s
					dK = 2.0*K
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
					
				return dK
					
		fun = _DerivativeFun(self, i)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
		
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					R = distance_matrix(X/l, Z/l, metric='euclidean')
					K = s*np.exp(-R)
					dK = np.zeros((m,n,d))
					R = R + 1e-16 #prevent division by zeros
					for i in xrange(d):
						dK[:,:,i] = 1.0/l[i]**2 * np.add.outer(Z[:,i],-X[:,i]) * K.T / R.T
				return dK
				
		return _DerivativeFun(self)
	
	def copy(self):
		d = self._d
		cp_kernel = ARDRBFKernel(self._params[0:d], self._params[d])
		return cp_kernel

class ARDRBFLinKernel(Kernel):
	'''
	Squared Exponential + linear covariance kernel with automatic relevance determinition
	distance measure. The kernel is parametrized as:
	
	k(x_p, x_q) = s**2 * exp(-(x_p-x_q)' * inv(L) *  (x_p-x_q)/ 2) 
	'''
	
	__slots__ = ('_d')
	
	def __init__(self, l, se=0.0, sl=0.0):
		l = np.asarray(l).ravel()
		params = np.r_[l,se,sl]
		Kernel.__init__(self, params)
		
		self._d = len(l)
		
	def __str__( self ):
		return "ARDSELinKernel()"
	
	def __call__(self, X, Z=None, diag=False):
		'''
		@todo: - optimize the distance computation of the diagonal dot prod
		'''
		
		xeqz = (Z == None)
			
		d = self._d
		l = np.exp(self.params[0:d])
		se = np.exp(2.0*self.params[d])
		sl = np.exp(2.0*self.params[d+1])
			
		X = np.dot(X, np.diag(1/l))
		if xeqz:	
			if diag:
				Ke = np.zeros(X.shape[0])
				Kl = np.sum(X*X,1)
			else:
				Ke = distance_matrix(X,  metric='euclidean')
				Kl = np.dot(X,X.T) 
		else:
			Z = np.dot(Z, np.diag(1.0/l))
			Ke = distance_matrix(X, Z, metric='euclidean')
			Kl = np.dot(X,Z.T)
			
		
		Ke = np.exp(-Ke)
		K = se*Ke + sl*Kl
		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		'''
		'''		
		xeqz = (Z == None)
			
		d = self.kernel._d
		l = np.exp(self.kernel.params[0:d])
		se = np.exp(2.0*self.kernel.params[d])
		sl = np.exp(2.0*self.kernel.params[d+1])
		
		grad = np.zeros(d+2)
			
		X = X/l
		if xeqz:
			if diag:
				R = np.zeros(X.shape[0])
				Kl = np.sum(X*X,1)
			else:
				R = distance_matrix(X, metric='euclidean')
				Kl = np.dot(X,X.T) 
		else:
			Z = Z/l
			R = distance_matrix(X,Z,metric='euclidean')
			Kl = np.dot(X,Z.T)
		
		Ke = se*np.exp(-R)
		Kl = sl*Kl
		
		#gradient of the length scales l
		R  = R+1e-16
		for i in xrange(d):
			x = X[:,i]
			if xeqz:
				if diag:
					Keprime = 0.0
					dKl = x*x
				else:
					Keprime = distance_matrix(x[:,np.newaxis], metric='sqeuclidean')
					dKl = np.outer(x,x.T)
			else:
				z = Z[:,i]
				Keprime = distance_matrix(x[:,np.newaxis], z[:,np.newaxis], metric='sqeuclidean')
				dKl = np.outer(x,z.T)
				
			dKe = Ke*Keprime/R
			dKl = -2.0*sl*dKl
			dK = dKe+dKl
			grad[i] = np.sum(covGrad*dK)

		#gradient of the signal variance of the squared-exp term
		dKse = 2.0 * Ke
		grad[d+1] = np.sum(covGrad*dKse)
		
		#gradient of the signal variance of the squared-exp term
		dKsl = 2.0 * Kl
		grad[d+2] = np.sum(covGrad*dKsl)

		return grad

			
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - check the gradient of the length scale vector (gradient check failed, 
				 but its identical to matlab gpml implementation)
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):	 
				xeqz = (Z == None)
					
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				se = np.exp(2.0*self.kernel.params[d])
				sl = np.exp(2.0*self.kernel.params[d+1])
					
				X = X/l
				if xeqz:
					if diag:
						R = np.zeros(X.shape[0])
						Kl = np.sum(X*X,1)
					else:
						R = distance_matrix(X, metric='euclidean')
						Kl = np.dot(X,X.T) 
				else:
					Z = Z/l
					R = distance_matrix(X,Z,metric='euclidean')
					Kl = np.dot(X,Z.T)
				
				Ke = se*np.exp(-R)
				Kl = sl*Kl
				if self.i < d:
					x = X[:,i]
					if xeqz:
						if diag:
							Keprime = 0.0
							dKl = x*x
						else:
							Keprime = distance_matrix(x[:,np.newaxis], metric='sqeuclidean')
							dKl = np.outer(x,x.T)
					else:
						z = Z[:,i]
						Keprime = distance_matrix(x[:,np.newaxis], z[:,np.newaxis], metric='sqeuclidean')
						dKl = np.outer(x,z.T)
						
					R  = R+1e-16
					dKe = Ke*Keprime/R
					dKl = -2.0*sl*dKl
					dK = dKe+dKl
				elif self.i == d:
					#gradient of the variance parameter s
					dK = 2.0*Ke
				elif self.i == d+1:
					dK = 2.0*Kl
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
					
				return dK
					
		fun = _DerivativeFun(self, i)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				se = np.exp(2.0*self.kernel.params[d])
				sl = np.exp(2.0*self.kernel.params[d+1])
		
		
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					#dK = np.zeros(n*d)
					dK = 2.0*np.diag(X)
				else:
					R = distance_matrix(X/l, Z/l, metric='euclidean')
					K = se*np.exp(-R)
					dKe = np.zeros((m,n,d))
					dKl = np.tile(Z[:,np.newaxis,:], (1,n,1))
					if xeqz:
						dKl[np.diag(np.ones(n,dtype=np.bool)),:] *= 2.0
					
					R = R + 1e-16 #prevent division by zeros
					for i in xrange(d):
						dKe[:,:,i] = 1.0/l[i]**2 * np.add.outer(Z[:,i],-X[:,i]) * K.T/R.T
						dKl[:,:,i] /= l[i]**2
					dK = dKe + sl*dKl
				return dK
			
			
				
		return _DerivativeFun(self)
	
	
	def copy(self):
		d = self._d
		cp_kernel = ARDRBFLinKernel(self._params[0:d], self._params[d], self._params[d+1])
		return cp_kernel


class SEKernel(Kernel):
	'''
	Squared Exponential covariance kernel with isotropic distance measure. 
	The kernel is parametrized as:
	
	k(x_p, x_q) = s**2 * exp(-|x_p-x_q|^2 / l^2) 
	'''
	
	def __init__(self, l, s=0.0):
		Kernel.__init__(self, np.asarray([l,s]))
		
	def __str__( self ):
		return "SqExpKernel({0},{1})".format(self.params[0],self.params[1])
	
	def __call__(self, X, Z=None, diag=False):
		
		l = np.exp(2.0*self.params[0])
		s = np.exp(2.0*self.params[1])
		xeqz = (Z == None)
		
		if xeqz and diag:
			R = np.zeros(X.shape[0])
		else:
			R = distance_matrix(X, Z, metric='sqeuclidean')
		#R = np.atleast_2d(R)
		K = s * np.exp(-R/(2.0*l))
		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		'''
		'''
		l = np.exp(2.0*self.params[0])
		s = np.exp(2.0*self.params[1])
		xeqz = (Z == None)
		
		grad = np.zeros(2)

		if xeqz and diag:
			R = np.zeros(X.shape[0])
		else:
			R = distance_matrix(X, Z, metric='sqeuclidean')
		K = s * np.exp(-R/(2.0*l))
		
		#gradient of the length scale l
		dKl = K * R/l
		grad[0] = np.sum(covGrad*dKl)
		
		#gradient of the signal variance
		dKs = 2.0 * K
		grad[1] = np.sum(covGrad*dKs)

		return grad
			
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				l = np.exp(2.0*self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
		
				xeqz = (Z == None)
		
				if xeqz and diag:
					R = np.zeros(X.shape[0])
				else:
					R = distance_matrix(X, Z, metric='sqeuclidean')
				K = s * np.exp(-R/(2.0*l))
				
				if self.i == 0:
					#gradient of scale parameter l
					dK = K * R/l 
				elif self.i == 1:
					#gradient of the variance parameter s
					dK = 2.0 * K
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				return dK
					
		fun = _DerivativeFun(self, i)
		return fun
	
	def derivateX(self):
								
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				l = np.exp(2.0*self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
		
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					R = distance_matrix(X, Z, metric='sqeuclidean')
					K = s * np.exp(-R/(2.0*l))
					dK = np.zeros((m,n,d))
					for i in xrange(d):
						dK[:,:,i] = 1.0/l * np.add.outer(Z[:,i],-X[:,i]) * K.T
						
				return dK
				
		return _DerivativeFun(self)
	
	def derivateXOld(self):
		class _DerivativeFunX(Kernel._IDerivativeFunX):
			def __init__(self, kernel):
				self.kernel = kernel
			
			def __call__(self, x, Z):
				x = np.squeeze(x)

				l = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
				
				R = distance_matrix(x, Z, metric='sqeuclidean')
				K = s * np.exp(-R/(2.0*(l**2)))
				
				d = len(x)
				G = np.zeros(Z.shape)
				for i in xrange(d):
					G[:,i] = 1.0/l**2 * (Z[:,i] - x[i]) * K
					
				return G
		fun = _DerivativeFunX(self)
		return fun


	
	def copy(self):
		cp_kernel = SEKernel(np.copy(self._params[0]), self._params[1])
		return cp_kernel
	
class ARDSEKernel(Kernel):
	'''
	Squared Exponential covariance kernel with automatic relevance determinition
	distance measure. The kernel is parametrized as:
	
	k(x_p, x_q) = s**2 * exp(-(x_p-x_q)' * inv(L) *  (x_p-x_q)/ 2) 
	'''
	
	__slots__ = ('_d')
	
	def __init__(self, l, s=0.0):
		l = np.asarray(l).ravel()
		params = np.r_[l,s]
		Kernel.__init__(self, params)
		
		self._d = len(l)
		
	def __str__( self ):
		return "ARDSEKernel()"
	
	def __call__(self, X, Z=None, diag=False):
		'''
		@todo: - optimize the distance computation of the diagonal dot prod
		'''
		
		xeqz = (Z == None)
			
		d = self._d
		l = np.exp(self.params[0:d])
		s = np.exp(2.0*self.params[d])
			
		X = X/l
		if xeqz:	
			if diag:
				K = np.zeros(X.shape[0])
			else:
				K = distance_matrix(X,  metric='sqeuclidean')
		else:
			Z = Z/l
			K = distance_matrix(X, Z, metric='sqeuclidean')
			
		
		K = s*np.exp(-K/2.0)
		return K
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		'''
		'''
		xeqz = (Z == None)
					
		d = self._d
		l = np.exp(self.params[0:d])
		s = np.exp(2.0*self.params[d])

		grad = np.zeros(d+1)

		X = X/l
		if xeqz:
			if diag:
				R = np.zeros(X.shape[0])
			else:
				R = distance_matrix(X, metric='sqeuclidean') 
		else:
			Z = Z/l
			R = distance_matrix(X, Z, metric='sqeuclidean')
		K = s*np.exp(-R/2)
		
		#gradient of the length scales l
		for i in xrange(d):
			if xeqz:
				if diag:
					Kprime = 0.0
				else:
					Kprime = distance_matrix(X[:,i,np.newaxis], metric='sqeuclidean')
			else:
				Kprime = distance_matrix(X[:,i,np.newaxis], 
										 Z[:,i,np.newaxis], metric='sqeuclidean')
						
			dKl = K*Kprime
			grad[i] = np.sum(covGrad*dKl)
		
		#gradient of the signal variance
		dKs = 2.0 * K
		grad[d] = np.sum(covGrad*dKs)

		return grad	
	
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - check the gradient of the length scale vector (gradient check failed, 
				 but its identical to matlab gpml implementation)
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):	 
				xeqz = (Z == None)
					
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
					
				
				X = X/l
				if xeqz:
					if diag:
						K = np.zeros(X.shape[0])
					else:
						K = distance_matrix(X,  metric='sqeuclidean') 
				else:
					Z = Z/l
					K = distance_matrix(X,Z,metric='sqeuclidean')
				
				K = s*np.exp(-K/2)
				if self.i < d:
					if xeqz:
						if diag:
							Kprime = 0.0
						else:
							Kprime = distance_matrix(X[:,i,np.newaxis], metric='sqeuclidean')
					else:
						Kprime = distance_matrix(X[:,i,np.newaxis], 
												 Z[:,i,np.newaxis], metric='sqeuclidean')
					dK = K*Kprime
				elif self.i == d:
					#gradient of the variance parameter s
					dK = 2.0*K
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
					
				return dK
					
		fun = _DerivativeFun(self, i)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
		
		
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					R = distance_matrix(X/l, Z/l, metric='sqeuclidean')
					K = s*np.exp(-R/2)
					dK = np.zeros((m,n,d))
					for i in xrange(d):
						dK[:,:,i] = 1.0/l[i]**2 * np.add.outer(Z[:,i],-X[:,i]) * K.T
				return dK
				
		return _DerivativeFun(self)
	
	def copy(self):
		d = self._d
		cp_kernel = ARDSEKernel(self._params[0:d], self._params[d])
		return cp_kernel


class ARDSELinKernel(Kernel):
	'''
	Squared Exponential + linear covariance kernel with automatic relevance determinition
	distance measure. The kernel is parametrized as:
	
	k(x_p, x_q) = s**2 * exp(-(x_p-x_q)' * inv(L) *  (x_p-x_q)/ 2) 
	'''
	
	__slots__ = ('_d')
	
	def __init__(self, l, se=0.0, sl=0.0):
		l = np.asarray(l).ravel()
		params = np.r_[l,se,sl]
		Kernel.__init__(self, params)
		
		self._d = len(l)
		
	def __str__( self ):
		return "ARDSELinKernel()"
	
	def __call__(self, X, Z=None, diag=False):
		'''
		@todo: - optimize the distance computation of the diagonal dot prod
		'''
		
		xeqz = (Z == None)
			
		d = self._d
		l = np.exp(self.params[0:d])
		se = np.exp(2.0*self.params[d])
		sl = np.exp(2.0*self.params[d+1])
			
		X = np.dot(X, np.diag(1/l))
		if xeqz:	
			if diag:
				Ke = np.zeros(X.shape[0])
				Kl = np.sum(X*X,1)
			else:
				Ke = distance_matrix(X,  metric='sqeuclidean')
				Kl = np.dot(X,X.T) 
		else:
			Z = np.dot(Z, np.diag(1.0/l))
			Ke = distance_matrix(X, Z, metric='sqeuclidean')
			Kl = np.dot(X,Z.T)
			
		
		Ke = np.exp(-Ke/2.0)
		K = se*Ke + sl*Kl
		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		'''
		'''
		xeqz = (Z == None)
					
		d = self.kernel._d
		l = np.exp(self.kernel.params[0:d])
		se = np.exp(2.0*self.kernel.params[d])
		sl = np.exp(2.0*self.kernel.params[d+1])

		grad = np.zeros(d+2)

		X = X/l
		if xeqz:
			if diag:
				Ke = np.zeros(X.shape[0])
				Kl = np.sum(X*X,1)
			else:
				Ke = distance_matrix(X, metric='sqeuclidean')
				Kl = np.dot(X,X.T) 
		else:
			Z = Z/l
			Ke = distance_matrix(X,Z,metric='sqeuclidean')
			Kl = np.dot(X,Z.T)		

		Ke = se*np.exp(-Ke/2)
		Kl = sl*Kl
			
		#gradient of the length scales l
		for i in xrange(d):
			x = X[:,i]
			if xeqz:
				if diag:
					Keprime = 0.0
					dKl = x*x
				else:
					Keprime = distance_matrix(x[:,np.newaxis], metric='sqeuclidean')
					dKl = np.outer(x,x.T)
			else:
				z = Z[:,i]
				Keprime = distance_matrix(x[:,np.newaxis], z[:,np.newaxis], metric='sqeuclidean')
				dKl = np.outer(x,z.T)
				
			dKe = Ke*Keprime
			dKl = -2.0*sl*dKl
			dK = dKe+dKl
			grad[i] = np.sum(covGrad*dK)
		
		#gradient of the signal variance of the squared-exp term
		dKse = 2.0 * Ke
		grad[d+1] = np.sum(covGrad*dKse)
		
		#gradient of the signal variance of the squared-exp term
		dKsl = 2.0 * Kl
		grad[d+2] = np.sum(covGrad*dKsl)

		return grad	

			
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - check the gradient of the length scale vector (gradient check failed, 
				 but its identical to matlab gpml implementation)
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):	 
				xeqz = (Z == None)
					
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				se = np.exp(2.0*self.kernel.params[d])
				sl = np.exp(2.0*self.kernel.params[d+1])
					
				X = X/l
				if xeqz:
					if diag:
						Ke = np.zeros(X.shape[0])
						Kl = np.sum(X*X,1)
					else:
						Ke = distance_matrix(X, metric='sqeuclidean')
						Kl = np.dot(X,X.T) 
				else:
					Z = Z/l
					Ke = distance_matrix(X,Z,metric='sqeuclidean')
					Kl = np.dot(X,Z.T)
				
				Ke = se*np.exp(-Ke/2)
				Kl = sl*Kl
				if self.i < d:
					x = X[:,i]
					if xeqz:
						if diag:
							Keprime = 0.0
							dKl = x*x
						else:
							Keprime = distance_matrix(x[:,np.newaxis], metric='sqeuclidean')
							dKl = np.outer(x,x.T)
					else:
						z = Z[:,i]
						Keprime = distance_matrix(x[:,np.newaxis], z[:,np.newaxis], metric='sqeuclidean')
						dKl = np.outer(x,z.T)
						
					dKe = Ke*Keprime
					dKl = -2.0*sl*dKl
					dK = dKe+dKl
				elif self.i == d:
					#gradient of the variance parameter s
					dK = 2.0*Ke
				elif self.i == d+1:
					dK = 2.0*Kl
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
					
				return dK
					
		fun = _DerivativeFun(self, i)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				se = np.exp(2.0*self.kernel.params[d])
				sl = np.exp(2.0*self.kernel.params[d+1])
		
		
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					#dK = np.zeros(n*d)
					dK = 2.0*np.diag(X)
				else:
					R = distance_matrix(X/l, Z/l, metric='sqeuclidean')
					K = se*np.exp(-R/2)
					dKe = np.zeros((m,n,d))
					dKl = np.tile(Z[:,np.newaxis,:], (1,n,1))
					if xeqz:
						dKl[np.diag(np.ones(n,dtype=np.bool)),:] *= 2.0
					for i in xrange(d):
						dKe[:,:,i] = 1.0/l[i]**2 * np.add.outer(Z[:,i],-X[:,i]) * K.T
						dKl[:,:,i] /= l[i]**2
					dK = dKe + sl*dKl
				return dK
			
			
				
		return _DerivativeFun(self)
	
	
	def copy(self):
		d = self._d
		cp_kernel = ARDSELinKernel(self._params[0:d], self._params[d], self._params[d+1])
		return cp_kernel


class GaussianKernel(Kernel):
	'''
	Squared Exponential covariance kernel with automatic relevance determinition
	distance measure. The kernel is parametrized as:
	
	k(x_p, x_q) = exp(-(x_p-x_q)' * inv(L) *  (x_p-x_q)/ 2) 
	'''
	
	__slots__ = ('_d')
	
	def __init__(self, l):
		params = np.asarray(l).ravel()
		Kernel.__init__(self, params)
		
		self._d = len(l)
		
	def __str__( self ):
		return "ARDSEKernel()"
	
	def __call__(self, X, Z=None, diag=False):
		'''
		@todo: - optimize the distance computation of the diagonal dot prod
		'''
		
		xeqz = (Z == None)
			
		d = self._d
		l = np.exp(self.params)
			
		P = np.diag(1/l)
		
		if xeqz:
			
			if diag:
				K = np.zeros(X.shape[0])
			else:
				K = distance_matrix(np.dot(P, X.T).T,  metric='sqeuclidean') 
		else:
			K = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, metric='sqeuclidean')
			
		K = np.exp(-K/2.0)
		detP = np.prod(1/(l**2.0))
		K = (2*np.pi)**(d/2.0) * np.sqrt(detP) * K
		#K = np.sqrt(detP) * K
		return K
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		'''
		'''
		xeqz = (Z == None)
			
		d = self.kernel._d
		l = np.exp(self.kernel.params)
		detP = np.prod(1/(l**2.0))

		grad = np.zeros(d)
		
		P = np.diag(1/l)
		if xeqz:
			if diag:
				R = np.zeros(X.shape[0])
			else:
				R = distance_matrix(np.dot(P, X.T).T,  metric='sqeuclidean') 
		else:
			R = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, 
								metric='sqeuclidean')
			
		K = np.exp(-R/2)
		K = (2*np.pi)**(d/2.0) * np.sqrt(detP) * K
		
		#gradient of the length scales l
		for i in xrange(d):
			if xeqz:
				if diag:
					Kprime = 0.0
				else:
					Kprime = distance_matrix(X[:,i,np.newaxis]/l[i], metric='sqeuclidean')
			else:
				Kprime = distance_matrix(X[:,i,np.newaxis]/l[i], 
										 Z[:,i,np.newaxis]/l[i], metric='sqeuclidean')
			dK = K*Kprime
			grad[i] = np.sum(covGrad*dK)

		return grad	

			
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - check the gradient of the length scale vector (gradient check failed, 
				 but its identical to matlab gpml implementation)
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):	 
				xeqz = (Z == None)
					
				d = self.kernel._d
				l = np.exp(self.kernel.params)
				detP = np.prod(1/(l**2.0))
				
				P = np.diag(1/l)
				if xeqz:
					if diag:
						R = np.zeros(X.shape[0])
					else:
						R = distance_matrix(np.dot(P, X.T).T,  metric='sqeuclidean') 
				else:
					R = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, 
										metric='sqeuclidean')
			
				K = np.exp(-R/2)
				K = (2*np.pi)**(d/2.0) * np.sqrt(detP) * K
				
				if self.i < d:
					#gradient of scale parameter l
					if xeqz:
						if diag:
							Kprime = 0.0
						else:
							Kprime = distance_matrix(X[:,i,np.newaxis]/l[i], metric='sqeuclidean')
					else:
						Kprime = distance_matrix(X[:,i,np.newaxis]/l[i], 
												 Z[:,i,np.newaxis]/l[i], metric='sqeuclidean')
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				
				dK = K*Kprime
				dK = dK - K
				return dK
					
		fun = _DerivativeFun(self, i)
		return fun

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
		
				P = np.diag(1/l)
	
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
				m = Z.shape[0]
		
				if xeqz and diag:
					dK = np.zeros(n*d)
				else:
					R = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, metric='sqeuclidean')
					K = np.exp(-R/2)
					detP = np.prod(1/(l**2.0))
					K = (2*np.pi)**(d/2.0) * np.sqrt(detP) * K

					dK = np.zeros((m,n,d))
					for i in xrange(d):
						dK[:,:,i] = 1.0/l[i]**2 * np.add.outer(Z[:,i],-X[:,i]) * K.T
				return dK
				
		return _DerivativeFun(self)


	
	def copy(self):
		cp_kernel = GaussianKernel(self._params)
		return cp_kernel

	
class LinearKernel(Kernel):
	'''
	todo: add a variance term
	'''
	def __init__(self):
		Kernel.__init__(self, np.array([]))
	
	def __str__(self):
		return 'Linear Kernel'
	
	def __call__(self, X, Z=None, diag=False):
		xeqz = (Z==None)
		if xeqz:
			if diag:
				K = np.sum(X*X,1)
			else:
				K = np.dot(X,X.T)
		else:
			K = np.dot(X,Z.T)
		return K
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		pass
	
	def derivate(self, i):
		'''
		@todo: return zero array
		'''
		pass
	
	def derivateXOld(self):
		'''
		'''
		class _DerivativeFunX(Kernel._IDerivativeFunX):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, x, Z):
				G = Z.copy()
				n = G.shape[0]
				for i in xrange(n):
					#todo: make diag part more clearly
					if np.all(np.equal(x, G[i])):
						G[i] = G[i]*2.0
				return G
							
		fun = _DerivativeFunX(self)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n = X.shape[0]
				if xeqz and diag:
					dK = 2.0*np.diag(X)
				else:
					dK = np.tile(Z[:,np.newaxis,:], (1,n,1))
					if xeqz:
						dK[np.diag(np.ones(n,dtype=np.bool)),:] *= 2.0
				return dK
				
		return _DerivativeFun(self)
	
	def copy(self):
		cp_kernel = LinearKernel()
		return cp_kernel

class BiasedLinearKernel(Kernel):
	
	def __init__(self, bias):
		Kernel.__init__(self, np.array([bias]))
	
	def __str__(self):
		return 'BiasedLinearKernel({0})'.format(self.params[0])
	
	def __call__(self, X, Z=None, diag=False):
		bias = np.exp(2.0*self.params[0])
		xeqz = (Z==None)
		
		if xeqz:
			if diag:
				K = np.sum(X*X,1)
			else:
				K = np.dot(X,X.T)
		else:
			K = np.dot(X,Z.T)
			
		K = (K+1.0)/bias 
		return K
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		'''
		'''
		bias = np.exp(2.0*self.kernel.params[0])
		xeqz = (Z==None)
				
		if xeqz:
			if diag:
				K = np.sum(X*X,1)
			else:
				K = np.dot(X,X.T)
		else:
			K = np.dot(X,Z.T)
		
		dK = -2.0*(K+1.0)/bias
		grad = np.zeros(1)
		grad[0] = np.sum(covGrad*dK)
		return grad

		
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				bias = np.exp(2.0*self.kernel.params[0])
				xeqz = (Z==None)
						
				if xeqz:
					if diag:
						K = np.sum(X*X,1)
					else:
						K = np.dot(X,X.T)
				else:
					K = np.dot(X,Z.T)
				
				
				dK = -2.0*(K+1.0)/bias
				return dK
			
		fun = _DerivativeFun(self)
		return fun
	
	def derivateXOld(self):
		'''
		'''
		class _DerivativeFunX(Kernel._IDerivativeFunX):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, x, Z):
				bias = np.exp(2.0*self.kernel.params[0])
				
				G = Z.copy()
				n = G.shape[0]
				for i in xrange(n):
					#todo: make diag part more clearly
					if np.all(np.equal(x, G[i])):
						G[i] = G[i]*2.0
				G = G/bias
				return G
							
		fun = _DerivativeFunX(self)
		return fun

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				bias = np.exp(2.0*self.kernel.params[0])
				
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n = X.shape[0]
				if xeqz and diag:
					dK = 2.0*np.diag(X)
				else:
					dK = np.tile(Z[:,np.newaxis,:], (1,n,1))
					if xeqz:
						dK[np.diag(np.ones(n,dtype=np.bool)),:] *= 2.0
				dK = dK/bias
				return dK
				
		return _DerivativeFun(self)

	
	def copy(self):
		cp_kernel = BiasedLinearKernel(self._params[0])
		return cp_kernel
	
class ARDLinearKernel(Kernel):
	
	__slots__ = ('_d')
	
	def __init__(self, l, s=0.0):
		l = np.asarray(l).ravel()
		params = np.r_[l,s]
		Kernel.__init__(self, params)
		
		self._d = len(l)
		
	def __str__(self):
		return 'ARDLinearKernel()'
	
	def __call__(self, X, Z=None, diag=False):
		d = self._d
		l = np.exp(self.params[0:d])
		s = np.exp(2.0*self.params[d])
		
		xeqz = (Z==None)
		
		X = X/l
		if xeqz:
			if diag:
				K = np.sum(X*X,1)
			else:
				K = np.dot(X,X.T)
		else:
			Z = Z/l
			K = np.dot(X,Z.T)
		
		K *= s
		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		'''
		'''
		d = self._d
		l = np.exp(self._params[0:d])
		s = np.exp(2.0*self._params[d])
		
		xeqz = (Z == None)
		
		grad = np.zeros(d+1)
		
		X = X/l
		
		#gradient of the length scales l
		for i in xrange(d):
			x = X[:,i]
			if xeqz:
				if diag:
					dK = x*x
				else:
					dK = np.outer(x,x.T)
			else:
				z = Z[:,i]/l[i]
				dK = np.outer(x,z.T)
			dKl = -2.0*s*dK
			grad[i] = np.sum(covGrad*dKl)
		
		#gradient of the signal variance
		if xeqz:
			if diag:
				K = np.sum(X*X,1)
			else:
				K = np.dot(X,X.T)
		else:
			Z = Z/l
			K = np.dot(X,Z.T)
		K = s*K

		dKs = 2.0 * K
		grad[d] = np.sum(covGrad*dKs)
		return grad

	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				i = self.i
				
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
				
				xeqz = (Z == None)
				
				X = X/l
				if i < d:
					x = X[:,i]
					if xeqz:
						if diag:
							dK = x*x
						else:
							dK = np.outer(x,x.T)
					else:
						Z = Z/l
						z = Z[:,i]
						dK = np.outer(x,z.T)
						
					dK = -2.0*s*dK
				elif i == d:
					
					if xeqz:
						if diag:
							K = np.sum(X*X,1)
						else:
							K = np.dot(X,X.T)
					else:
						Z = Z/l
						K = np.dot(X,Z.T)
					K = s*K
					
					dK = 2.0*K
				else:
					raise TypeError('Unknown hyperparameter')
				
				#K = -2*K
				return dK
			
		fun = _DerivativeFun(self, i)
		return fun

	def derivateXOld(self):
		'''
		'''
		class _DerivativeFunX(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, x, Z):
				d = self.kernel._d
				l = np.exp(2.0*self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
				
				G = Z.copy()
				n = G.shape[0]
				for i in xrange(n):
					#todo: make diag part more clearly
					if np.all(np.equal(x, G[i])):
						G[i] = G[i]*2.0
				G = s * G*1/l
				return G
							
		fun = _DerivativeFunX(self)
		return fun
	
	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				d = self.kernel._d
				l = np.exp(2.0*self.kernel.params[0:d])
				s = np.exp(2.0*self.kernel.params[d])
				
				xeqz = (Z == None)
				if xeqz:
					Z = X
				
				n,d = X.shape
		
				if xeqz and diag:
					dK = 2.0*np.diag(X)
				else:
					dK = np.tile(Z[:,np.newaxis,:], (1,n,1))
					if xeqz:
						dK[np.diag(np.ones(n,dtype=np.bool)),:] *= 2.0
					#TODO: optimize loop
					for i in xrange(d):
						dK[:,:,i] /= l[i]
				dK = dK*s
				
				return dK

		return _DerivativeFun(self)

	
	def copy(self):
		d = self._d
		cp_kernel = ARDLinearKernel(np.copy(self._params[0:d]), self._params[d])
		return cp_kernel
	

class PolynomialKernel(Kernel):
	
	__slots__ = ('_degree')
	
	def __init__(self, degree, c, s):
		Kernel.__init__(self, np.array([c,s]))
		self._degree = degree
		
	def __str__(self):
		return 'PolynomialKernel({0},{1})'.format(self.params[0],self.params[1])
	
	def __call__(self, X, Z=None, diag=False):
		d = self._degree
		c = np.exp(self.params[0])
		s = np.exp(2.0*self.params[1])
		xeqz = (Z==None)
		
		if xeqz: 
			if diag:
				K = np.sum(X*X,1)			
			else:
				K = np.dot(X,X.T)
		else:
			K = np.dot(X,Z.T)
			
		K = s * (K + c)**d
		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		d = self.kernel._degree
		c = np.exp(self.kernel.params[0])
		s = np.exp(2.0*self.kernel.params[1])
		
		xeqz = (Z==None)
		
		grad = np.zeros(2)
		if xeqz: 
			if diag:
				K = np.sum(X*X,1)
			else:
				K = np.dot(X,X.T)
		else:
			K = np.dot(X,Z.T)

		#gradient of bias parameter c
		dKc = s * c * d * (K + c)**(d-1)
		grad[0] = np.sum(covGrad*dKc)
		#gradient for the signal variance
		dKs = 2.0*s * (K + c)**d 
		grad[1] = np.sum(covGrad*dKs)
		return grad
	
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				d = self.kernel._degree
				c = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
				
				xeqz = (Z==None)
		
				if xeqz: 
					if diag:
						K = np.sum(X*X,1)
					else:
						K = np.dot(X,X.T)
				else:
					K = np.dot(X,Z.T)
								
				if self.i == 0:
					#gradient of bias parameter c
					K = s * c * d * (K + c)**(d-1)
		
				elif self.i == 1:
					K = 2.0*s * (K + c)**d 
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				return K
		
		fun = _DerivativeFun(self, i)
		return fun
	
	def derivateXOld(self):
		'''
		'''
		class _DerivativeFunX(Kernel._IDerivativeFunX):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, x, Z):
				d = self.kernel._degree
				c = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
				
				K = (np.dot(Z, x) + c)**(d-1)
				
				m = len(x)
				G = np.zeros(Z.shape)
				for i in xrange(m):
					G[:,i] = d*s*Z[:,i]*K

				n = len(Z)
				for i in xrange(n):
					#todo: make diag part more clearly
					if np.all(np.equal(x, Z[i])):
						G[i] = G[i]*2.0
				
				return G
							
		fun = _DerivativeFunX(self)
		return fun

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				d = self.kernel._degree
				c = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
			
				xeqz = (Z == None)
				if xeqz:
					Z = X
					
				if xeqz and diag:
					K = np.sum(X*X,1)
				else:
					K = np.dot(X,Z.T)
					
				K = s*d*(K+c)**(d-1)
				
				n,d = X.shape
				if xeqz and diag:
					dK = 2.0*np.diag(X)
					dK = dK*K
				else:
					dK = np.tile(Z[:,np.newaxis,:], (1,n,1))
					if xeqz:
						dK[np.diag(np.ones(n,dtype=np.bool)),:] *= 2.0
					#TODO: optimize loop
					for i in xrange(d):
						dK[:,:,i] *= K.T 
				
				return dK
				
		return _DerivativeFun(self)

	
	def copy(self):
		cp_kernel = PolynomialKernel(self._degree, self._params[0], self._params[1])
		return cp_kernel

class ARDPolynomialKernel(Kernel):
	
	__slots__ = ('_degree',
				 '_d')
	
	def __init__(self, degree, l, c, s):
		l = np.asarray(l).ravel()
		params = np.r_[l,c,s]
		Kernel.__init__(self, params)
		
		self._d = len(l)
		self._degree = degree
		
	def __str__(self):
		return 'ARDPolynomialKernel({0},{1})'.format(self.params[0],self.params[1])
	
	def __call__(self, X, Z=None, diag=False):
		deg = self._degree
		d = self._d
		l = np.exp(self.params[0:d])
		c = np.exp(self.params[d])
		s = np.exp(2.0*self.params[d+1])
		
		xeqz = (Z==None)
		
		X = X/l
		if xeqz: 
			if diag:
				K = np.sum(X*X,1)			
			else:
				K = np.dot(X,X.T)
		else:
			Z = Z/l
			K = np.dot(X,Z.T)
			
		K = s * (K + c)**deg
		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		deg = self.kernel._degree
		d = self.kernel._d
		l = np.exp(self.kernel.params[0:d])
		c = np.exp(self.kernel.params[d])
		s = np.exp(2.0*self.kernel.params[d+1])
		
		xeqz = (Z==None)
		
		grad = np.zeros(d+2)
		X/l
		if xeqz: 
			if diag:
				K = np.sum(X*X,1)
			else:
				K = np.dot(X,X.T)
		else:
			Z = Z/l
			K = np.dot(X,Z.T)

		#gradients for the length scales
		for i in xrange(d):
			x = X[:,i]
			if xeqz:
				if diag:
					dK = x*x
				else:
					dK = np.outer(x,x.T)
			else:
				z = Z[:,i]
				dK = np.outer(x,z.T)
				
			dKl = -2*s*deg*dK * (K+c)**(deg-1)
			grad[i] = np.sum(covGrad*dKl)

		#gradient of bias parameter c
		dKc = s * c * deg * (K + c)**(deg-1)
		grad[d] = np.sum(covGrad*dKc)
		#gradient for the signal variance
		dKs = 2.0*s * (K + c)**deg
		grad[d+1] = np.sum(covGrad*dKs)
		return grad


	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				deg = self.kernel._degree
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				c = np.exp(self.kernel.params[d])
				s = np.exp(2.0*self.kernel.params[d+1])
				
				xeqz = (Z==None)
		
				X = X/l
				if xeqz: 
					if diag:
						K = np.sum(X*X,1)
					else:
						K = np.dot(X,X.T)
				else:
					Z = Z/l
					K = np.dot(X,Z.T)
					
				if self.i < d:
					x = X[:,i]
					if xeqz:
						if diag:
							dK = x*x
						else:
							dK = np.outer(x,x.T)
					else:
						z = Z[:,i]
						dK = np.outer(x,z.T)
						
					dK = -2*s*deg*dK * (K+c)**(deg-1)

				elif self.i == d:
					#gradient of bias parameter c
					dK = s * c * deg * (K + c)**(deg-1)
		
				elif self.i == d+1:
					dK = 2.0*s * (K + c)**deg
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				return dK
		
		fun = _DerivativeFun(self, i)
		return fun
	

	def derivateX(self):
		
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def __call__(self, X, Z=None, diag=False):
				deg = self.kernel._degree
				d = self.kernel._d
				l = np.exp(self.kernel.params[0:d])
				c = np.exp(self.kernel.params[d])
				s = np.exp(2.0*self.kernel.params[d+1])
			
				xeqz = (Z == None)
				if xeqz:
					Z = X
					
				if xeqz and diag:
					K = np.sum(X*X,1)
				else:
					K = np.dot(X,Z.T)
					
				K = s*deg*(K+c)**(deg-1)
				
				n,d = X.shape
				if xeqz and diag:
					dK = 2.0*np.diag(X)
					dK = dK*K
				else:
					dK = np.tile(Z[:,np.newaxis,:], (1,n,1))
					if xeqz:
						dK[np.diag(np.ones(n,dtype=np.bool)),:] *= 2.0
					#TODO: optimize loop
					for i in xrange(d):
						dK[:,:,i] *= K.T/l[i]**2 
				
				return dK
				
		return _DerivativeFun(self)

	
	def copy(self):
		cp_kernel = PolynomialKernel(self._degree, self._params[0], self._params[1])
		return cp_kernel

	
class PiecewisePolyKernel(Kernel):
	
	__slots__ = ('v'	#degree of the polynom
				 )
	pass

class RQKernel(Kernel):
	pass

class MaternKernel(Kernel):
	'''
	@todo: - implement derivate wrt X
	'''
	__slots__ = ('_degree',
				 '_m',
				 '_dm')
	
	def __init__(self, degree, l, s):
		Kernel.__init__(self, np.array([l,s]))
		self._degree = degree
		
		if degree == 1:
			f = lambda t: 1
			df = lambda t: 1
			pass
		elif degree == 3:
			f = lambda t: 1 + t
			df = lambda t: t
			pass
		elif degree == 5:
			f = lambda t: 1 + t*(1.0+t/3.0)
			df = lambda t: t*(1.0+t)/3.0
		else:
			raise ValueError('degree must be 1, 3 or 5.')
		
		m = lambda t: f(t)*np.exp(-t)
		dm = lambda t: df(t)*t*np.exp(-t)
		
		self._m = m
		self._dm = dm 
		
		
	def __str__(self):
		return 'MaternKernel()'
	
	def __call__(self, X, Z=None, diag=False):
		
		d = self._degree
		m = self._m
		l = np.exp(self.params[0])
		s = np.exp(2.0*self.params[1])
		
		xeqz = (Z==None)
		if xeqz:
			if diag:
				R = np.zeros(X.shape[0])
			else:
				R = distance_matrix(X, metric='euclidean')
		else:
			R = distance_matrix(X, Z, metric='euclidean')
		
		K = np.sqrt(d)*R/l   
		K = s*m(K)
		return K

	def gradient(self, covGrad, X, Z=None, diag=False):
		d = self.kernel._degree
		m = self.kernel._m
		dm = self.kernel._dm

		l = np.exp(self.kernel.params[0])
		s = np.exp(2.0*self.kernel.params[1])
		
		xeqz = (Z==None)
		if xeqz:
			if diag:
				R = np.zeros(X.shape[0])
			else:
				R = distance_matrix(X, metric='euclidean')
		else:
			R = distance_matrix(X, Z, metric='euclidean')
			
		K = np.sqrt(d)*R/l	   
		
		grad = np.zeros(2)
		#todo: has big error for high variance or length scale
		dKl = s*dm(K)
		grad[0] = np.sum(covGrad*dKl)
		
		dKs = 2.0*s * m(K)
		grad[1] = np.sum(covGrad*dKs)
		
		return grad
			

	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				
				d = self.kernel._degree
				m = self.kernel._m
				dm = self.kernel._dm
				
				l = np.exp(self.kernel.params[0])
				s = np.exp(2.0*self.kernel.params[1])
				
				xeqz = (Z==None)
				if xeqz:
					if diag:
						R = np.zeros(X.shape[0])
					else:
						R = distance_matrix(X, metric='euclidean')
				else:
					R = distance_matrix(X, Z, metric='euclidean')
					
				K = np.sqrt(d)*R/l	   
				if self.i == 0:
					#todo: has big error for high variance or length scale
					K = s*dm(K)
				elif self.i == 1:
					K = 2.0*s * m(K)
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				return K
		
		fun = _DerivativeFun(self, i)
		return fun

	def derivateX(self, i):
		raise NotImplementedError()


	def copy(self):
		cp_kernel = MaternKernel(self._degree, self._params[0], self._params[1])
		return cp_kernel


class NeuralNetKernel(Kernel):
	pass

class PeriodicKernel(Kernel):
	'''
	@todo: - implement derivate wrt to X
	'''
	def __init__(self, l, p, s=0.0):
		Kernel.__init__(self, np.asarray([l,p,s]))

	def __str__( self ):
		return "PeriodicKernel()"
	
	def __call__(self, X, Z=None, diag=False):
		
		l = np.exp(self.params[0])
		p = np.exp(self.params[1])
		s = np.exp(2.0*self.params[2])

		xeqz = (Z==None)
		if xeqz:
			if diag:
				R = np.zeros(X.shape[0])
			else:
				R = distance_matrix(X, metric='euclidean')
		else:
			R = distance_matrix(X, Z, metric='euclidean')
		
		K = np.pi*R/p
		K = (np.sin(K)/l)**2
		K = s * np.exp(-2.0*K)
		return K
	
	def gradient(self, covGrad, X, Z=None, diag=False):
		l = np.exp(self.kernel.params[0])
		p = np.exp(self.kernel.params[1])
		s = np.exp(2.0*self.kernel.params[2])

		xeqz = (Z==None)
		if xeqz:
			if diag:
				R = np.zeros(X.shape[0])
			else:
				R = distance_matrix(X, metric='euclidean')
		else:
			R = distance_matrix(X, Z, metric='euclidean')
		
		K = np.pi*R/p

		grad = np.zeros(3)
		
		Ktmp = np.sin(K)**2
		dKl = 4.0*s * Ktmp * np.exp(-2.0*Ktmp/l**2) / l**2
		grad[0] = np.sum(covGrad*dKl)
		
		R = 4.0*s * np.pi*R
		Ktmp = R * np.cos(K) * np.sin(K) * np.exp(-2.0*(np.sin(K)/l)**2)
		dKp = Ktmp/(l**2 * p)
		grad[1] = np.sum(covGrad*dKp)
		
		Ktmp = (np.sin(K)/l)**2
		dKs = 2.0*s * np.exp(-2.0*Ktmp)
		grad[2] = np.sum(covGrad*dKs)
		
		return grad
		

	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		class _DerivativeFun(Kernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
				
			def __call__(self, X, Z=None, diag=False):
				l = np.exp(self.kernel.params[0])
				p = np.exp(self.kernel.params[1])
				s = np.exp(2.0*self.kernel.params[2])
				
				xeqz = (Z==None)
				if xeqz:
					if diag:
						R = np.zeros(X.shape[0])
					else:
						R = distance_matrix(X, metric='euclidean')
				else:
					R = distance_matrix(X, Z, metric='euclidean')
				
				K = np.pi*R/p
				if self.i == 0:
					K = np.sin(K)**2
					K = 4.0*s * K * np.exp(-2.0*K/l**2) / l**2
				elif self.i == 1:
					R = 4.0*s * np.pi*R
					K = R * np.cos(K) * np.sin(K) * np.exp(-2.0*(np.sin(K)/l)**2)
					K = K/(l**2 * p)
				elif self.i == 2:
					K = (np.sin(K)/l)**2
					K = 2.0*s * np.exp(-2.0*K)
				else:
					#@todo: - create a passende exception
					raise TypeError('Unknown hyperparameter')
				
				return K
					
		fun = _DerivativeFun(self, i)
		return fun

	def derivateX(self, i):
		raise NotImplementedError()

	def copy(self):
		cp_kernel = PeriodicKernel(self._params[0], self._params[1], self._params[2])
		return cp_kernel

class ConvolvedKernel(Kernel):
	
	__slots__ = ('_ntheta'
				)
	
	def __init__(self, params, ntheta=0):
		'''
		'''
		Kernel.__init__(self, params)
		self._ntheta = ntheta
	
		
	def __call__(self, X, Z=None, thetaP=None, thetaQ=None, diag=False, latent=False):
		'''
		Computes the covariance matrix cov[f_p(x), f_p(x')] of the convolved process f_p
		Computes the cross covariance matrix cov[f_p(x), f_q(x')] of the convolved process f_p and f_q
		Computes the cross covariance matrix cov[f_p(x), u(x')] of the convolved process f_p and the latent process u
		'''
		xeqz = (Z==None)
		
		if xeqz:
			if latent:
				K = self.lcov(X, diag=diag)
			else:
				K = self.cov(X, thetaP, diag)
		else:
			if latent and thetaP==None and thetaQ==None:
				K = self.lcov(X, Z, diag)
			else:
				K = self.ccov(X, Z, thetaP, thetaQ, latent)
			
		return K
	
	@abstractmethod
	def cov(self, X, theta, diag=False):
		'''
		'''

	@abstractmethod
	def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		Computes the cross covariance matrix cov[f_p(x), f_q(x)]
		Computes the cross covariance matrix cov[f_p(x), f_q(x)]
		'''
		
	@abstractmethod
	def lcov(self, X, Z=None, diag=False):
		'''
		Computes the covariance matrix cov[u(x), u(x')] uof the latent process u
		'''
	
	@abstractmethod
	def derivateTheta(self, i):
		'''
		Returns the derivateve of the smoothing kernel parameters
		'''
	
	@abstractmethod
	def derivateX(self):
		'''
		'''
		
	def _number_of_theta(self):
		return self._ntheta
	
	ntheta = property(fget=_number_of_theta)
	
	
	def copy(self):
		params = np.copy(self._params)
		new_kernel = self.__class__(params)
		new_kernel._ntheta = self._ntheta
		return new_kernel
			
	class _IDerivativeFun(object):

		def __call__(self, X, Z=None, thetaP=None, thetaQ=None, diag=False, latent=False):
		
			xeqz = (Z==None)
			
			if xeqz:
				if latent:
					K = self.lcov(X, diag)
				else:
					K = self.cov(X, thetaP, diag)
			else:
				if latent and thetaP==None and thetaQ==None:
					K = self.lcov(X, Z, diag)
				else:
					K = self.ccov(X, Z, thetaP, thetaQ, latent)
				
				
			return K


		@abstractmethod
		def cov(self, X, theta, diag=False):
			'''
			'''
		
		@abstractmethod
		def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
			'''
			'''
		
		@abstractmethod
		def lcov(self, X, Z=None, diag=False):
			'''
			'''

	class _IDerivativeFunX(object):

		def __call__(self, x, Z, thetaP=None, thetaQ=None, latent=False):
		
			xeqz = (thetaP==None or (thetaQ==None and latent==False))

			if xeqz:
				if latent:
					K = self.lcov(x, Z)
				else:
					K = self.cov(x, Z, thetaP)
			else:
				K = self.ccov(x, Z, thetaP, thetaQ, latent)
				
			return K


		@abstractmethod
		def cov(self, x, Z, theta):
			'''
			'''
		
		@abstractmethod
		def ccov(self, x, Z, thetaP, thetaQ=None, latent=False):
			'''
			'''
		
		@abstractmethod
		def lcov(self, x, Z):
			'''
			'''

class DiracConvolvedKernel(ConvolvedKernel):

	__slots__ = ('_kernel'	#latent kernel
				)

	def __init__(self, kernel):
		'''
		'''
		params = kernel.params
		self._kernel = kernel
		self._params = params
		self._n = len(params)
		self._ntheta = 1


	def cov(self, X, theta, diag=False):
		'''
		'''
		kernel = self._kernel
		s = np.exp(2*theta[0])
		K = kernel(X, diag=diag)
		K = s*K
		return K
	
	def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		Computes the cross covariance matrix cov[f_p(x), f_q(x')]
		Computes the cross covariance matrix cov[f_p(x), u(x')]
		'''
		kernel = self._kernel
		K = kernel(X,Z)
		sp = np.exp(thetaP[0])
		K = sp*K
		
		if not latent:
			sq = np.exp(thetaQ[0])
			K = sq*K
			
		return K
		
	def lcov(self, X, Z=None, diag=False):
		'''
		Computes the covariance matrix cov[u(x), u(x')] uof the latent process u
		'''	
		kernel = self._kernel
		K = kernel(X, Z, diag=diag)
		return K
	
	def gradient_cov(self, covGrad, X, theta, diag=False):
		'''
		'''
		kernel = self._kernel
		s = np.exp(2*theta[0])
		grad = s*kernel.gradient(covGrad, X, diag=diag)
		
		dK = 2.0*s*kernel(X, diag=diag)
		gradTheta = np.array([np.sum(covGrad*dK)])
		
		return grad, gradTheta
	
	def gradient_ccov(self, covGrad, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		'''
		kernel = self._kernel
		sp = np.exp(thetaP[0])
		grad = sp*kernel.gradient(covGrad, X, Z)
		gradP = sp*np.sum(covGrad*kernel(X,Z))
		
		if not latent:
			sq = np.exp(thetaQ[0])
			grad = sq*grad
			gradP = sq*gradP
			gradQ = sq*gradP 
			return grad, gradP, gradQ
		
		return grad, gradP
		
	def gradient_lcov(self, covGrad, X, Z=None, diag=False):
		kernel = self._kernel
		return kernel.gradient(covGrad, X, Z, diag=diag)
		
	def derivate(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.dKernel = kernel.derivate(i)
			
			def cov(self, X, theta, diag=False):
				dKernel = self.dKernel
				s = np.exp(2.0*theta[0])
				dK = s*dKernel(X, diag=diag)
				return dK
			
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				dKernel = self.dKernel
				
				dK = dKernel(X,Z)
				sp = np.exp(thetaP[0])
				dK = sp*dK
				
				if not latent:
					sq = np.exp(thetaQ[0])
					dK = sq*dK
					
				return dK
			
			def lcov(self, X, Z=None, diag=False):	
				dKernel = self.dKernel
				dK = dKernel(X,Z,diag=diag)
				return dK

								
		fun = _DerivativeFun(self._kernel, i)
		return fun

	def derivateTheta(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.dKernel = kernel.derivate(i)
			
			def cov(self, X, theta, diag=False):
				kernel = self.kernel
				s = np.exp(2.0*theta[0])
				dK = 2.0*s*kernel(X, diag=diag)
				return dK
			
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				kernel = self.kernel
				
				sp = np.exp(thetaP[0])
				K = kernel(X,Z)
				dK = sp*K
				if not latent:
					sq = np.exp(thetaQ[0])
					dKp = sq*dK
					dKq = sq*dK
					return dKp, dKq
					
				return dK
			
			
			def lcov(self, X, Z=None, diag=False):
				raise NotImplementedError('latent cov has no hyperparameter theta')

		if i != 0:
			raise TypeError('Unknown hyperparameter')
			
								
		fun = _DerivativeFun(self._kernel, i)
		return fun

	def derivateX(self):
		'''
		'''
		
		class _DerivativeFunX(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel):
				self.dKernelX = kernel.derivateX()
				
			def cov(self, X, theta, diag=False):
				dKernelX = self.dKernelX
				s = np.exp(2*theta[0])
				dK = s*dKernelX(X, diag=diag)
				return dK

			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				dKernelX = self.dKernelX
				
				sp = np.exp(thetaP[0])
				dK = sp*dKernelX(X,Z)
				
				if not latent:
					sq = np.exp(thetaQ[0])
					dK = sq*dK
					
				return dK
			
			def lcov(self, X, Z=None, diag=False):
				dKernelX = self.dKernelX
				dK = dKernelX(X,Z,diag)
				return dK
							
		fun = _DerivativeFunX(self._kernel)
		return fun
	
	def copy(self):
		cp_kernel = DiracConvolvedKernel(self._kernel.copy())
		return cp_kernel

class ExpGaussianKernel(ConvolvedKernel):

	__slots__ = ()

	def __init__(self, l):
		'''
		'''
		#l = np.asarray(l).ravel()
		ConvolvedKernel.__init__(self, l, 2)


	def cov(self, X, theta, diag=False):
		'''
		'''
		l = np.exp(2.0*self.params[0])
		lp = 2.0*np.exp(2.0*theta[0])
		sp = np.exp(2.0*theta[1])
		
		lpu = l + lp
		K = self._compute_gausskern(lpu, X, diag=diag)
		K = sp * np.sqrt(l/lpu) * K
		
		return K
	
	def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		Computes the cross covariance matrix cov[f_p(x), f_q(x')]
		Computes the cross covariance matrix cov[f_p(x), u(x')]
		'''
		l = np.exp(2.0*self.params[0])
		lp = np.exp(2.0*thetaP[0])
		sp = np.exp(thetaP[1])
		
		if latent:
			'''
			Compute cov[f_p(x), u(x')]
			'''
			lpu = l + lp	
			K = self._compute_gausskern(lpu, X, Z)
			#K = sp * np.sqrt(detLpu/detL) * K
			K = sp * np.sqrt(l/lpu) * K

		else:
			'''
			Compute cov[f_p(x), f_q(x')]
			'''			
			lq = np.exp(2*thetaQ[0])
			sq = np.exp(thetaQ[1])
			
			lpq = l+lp+lq
			K = self._compute_gausskern(lpq, X, Z)
			#K = sp * sq * np.sqrt(detLpq/detL) * K
			K = sp * sq * np.sqrt(l/lpq) * K
		
		return K			
		
	def lcov(self, X, Z=None, diag=False):
		'''
		Computes the covariance matrix cov[u(x), u(x')] uof the latent process u
		'''		
		l = np.exp(2.0*self.params[0])
		K = self._compute_gausskern(l, X, Z, diag=diag)
		return K
	
	def gradient_cov(self, covGrad, X, theta, diag=False):
		'''
		'''
		l = np.exp(2.0*self.params[0])
		lp = 2.0*np.exp(2.0*theta[0])
		sp = np.exp(2.0*theta[1])
		
		lpu = l + lp
		dl = l/lpu**2
		dlp = lp/lpu**2
		
		K,R = self._compute_gausskern(lpu, X, diag=diag, retR=True)	
		dKl = K*R*dl
		dKlp = K*R*dlp
		
		dnorm_l = l/np.sqrt(lpu) * (1.0/np.sqrt(l)-np.sqrt(l)/lpu)
		dnorm_lp = -np.sqrt(l)*lp/lpu**1.5
		dKl = np.sqrt(l/lpu) * dKl + dnorm_l * K
		dKlp = np.sqrt(l/lpu) * dKlp + dnorm_lp * K
		
		grad = sp*np.sum(covGrad*dKl)
		
		gradTheta = np.zeros(2)
		gradTheta[0] = sp*np.sum(covGrad*dKlp)
		gradTheta[1] = 2.0*sp*np.sqrt(l/lpu)*np.sum(covGrad*K)
		
		return grad, gradTheta
	
	def gradient_ccov(self, covGrad, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		'''
		'''
		'''
		l = np.exp(2.0*self.params[0])
		lp = np.exp(2.0*thetaP[0])
		sp = np.exp(thetaP[1])
		
		
		if latent:
			lpu = l + lp
			dl = l/lpu**2
			dlp = lp/lpu**2
			
			K,R = self._compute_gausskern(lpu, X, Z,  retR=True)
				
			dKl = K*R*dl
			dKp = K*R*dlp
			
			dnorm_l = l/np.sqrt(lpu) * (1.0/np.sqrt(l)-np.sqrt(l)/lpu)
			dnorm_p = -np.sqrt(l)*lp/lpu**1.5
			dKl = np.sqrt(l/lpu) * dKl + dnorm_l * K
			dKp = np.sqrt(l/lpu) * dKp + dnorm_p * K
			
			grad = sp*np.sum(covGrad*dKl)
			
			gradP = np.zeros(2)
			gradP[0] = sp*np.sum(covGrad*dKp)
			gradP[1] = sp*np.sqrt(l/lpu)*np.sum(covGrad*K)
			
			return grad, gradP					
		else:
			lq = np.exp(2.0*thetaQ[0])
			sq = np.exp(thetaQ[1])
			
			lpq = l + lp + lq
			dl = l/lpq**2
			dlp = lp/lpq**2
			dlq = lq/lpq**2
			
			
			K,R = self._compute_gausskern(lpq, X, Z,  retR=True)
				
			dKl = K*R*dl
			dKp = K*R*dlp
			dKq = K*R*dlq
			
			dnorm_l = l/np.sqrt(lpq) * (1.0/np.sqrt(l)-np.sqrt(l)/lpq)
			dnorm_p = -np.sqrt(l)*lp/lpq**1.5
			dnorm_q = -np.sqrt(l)*lq/lpq**1.5
			dKl = np.sqrt(l/lpq) * dKl + dnorm_l * K
			dKp = np.sqrt(l/lpq) * dKp + dnorm_p * K
			dKq = np.sqrt(l/lpq) * dKq + dnorm_q * K
			
			grad = sp*sq*np.sum(covGrad*dKl)
			
			gradP = np.zeros(2)
			gradP[0] = sp*sq*np.sum(covGrad*dKp)
			gradP[1] = sp*sq*np.sqrt(l/lpq)*np.sum(covGrad*K)
			
			gradQ = np.zeros(2)
			gradQ[0] = sp*sq*np.sum(covGrad*dKq)
			gradQ[1] = sp*sq*np.sqrt(l/lpq)*np.sum(covGrad*K)
			
			return grad, gradP, gradQ					

	def gradient_lcov(self, covGrad, X, Z=None, diag=False):
		l = np.exp(2.0*self.params[0])
		K,R = self._compute_gausskern(l, X, Z, diag=diag, retR=True)
		dK = K*R/l
		grad = np.array([np.sum(covGrad*dK)])
		return grad


	def derivate(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
			
			def cov(self, X, theta, diag=False):
				l = np.exp(2.0*self.kernel.params[0])
				lp = 2.0*np.exp(2.0*theta[0])
				sp = np.exp(2.0*theta[1])
				
				lpu = l + lp
				dl = l/lpu**2
				
				K,R = self.kernel._compute_gausskern(lpu, X, diag=diag, retR=True)	
				dK = K*R*dl
				dnorm = l/np.sqrt(lpu) * (1.0/np.sqrt(l)-np.sqrt(l)/lpu)
				dK = np.sqrt(l/lpu) * dK + dnorm * K
				dK = sp * dK
				
				return dK
			
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				l = np.exp(2.0*self.kernel.params[0])
				lp = np.exp(2.0*thetaP[0])
				sp = np.exp(thetaP[1])
				
				if latent:
					'''
					Compute cov[f_p(x), u(x')]
					'''
					lpu = l + lp
					dl = l/lpu**2
					
					K,R = self.kernel._compute_gausskern(lpu, X, Z, retR=True)
					dK = K*R*dl
					#dnorm = detLpui/np.sqrt(detL*detLpu) - np.sqrt(detLpu/detL) 
					#dK = np.sqrt(detLpu/detL) * dK + dnorm * K
					dnorm = l/np.sqrt(lpu) * (1.0/np.sqrt(l)-np.sqrt(l)/lpu)
					dK = np.sqrt(l/lpu) * dK + dnorm * K
					dK = sp * dK
		
				else:
					'''
					Compute cov[f_p(x), f_q(x')]
					'''
					lq = np.exp(2.0*thetaQ[0])
					sq = np.exp(thetaQ[1])
					
					lpq = l+lp+lq
					dl = l/lpq**2
					
					K,R = self.kernel._compute_gausskern(lpq, X, Z, retR=True)
					dK = K*R*dl
					#dnorm = detLpqi/np.sqrt(detL*detLpq) - np.sqrt(detLpq/detL)
					#dK = np.sqrt(detLpq/detL) * dK + dnorm * K
					dnorm = l/np.sqrt(lpq) * (1.0/np.sqrt(l)-np.sqrt(l)/lpq) 
					dK = np.sqrt(l/lpq) * dK + dnorm * K
					dK = sp * sq * dK
			
				return dK			
			
			def lcov(self, X, Z=None, diag=False):	
				l = np.exp(2.0*self.kernel.params[0])
				K,R = self.kernel._compute_gausskern(l, X, Z, diag=diag, retR=True)
				dK = K*R/l
				
				return dK

		if i > 1:	
			raise TypeError('Unknown hyperparameter: {0}'.format(i))
								
		fun = _DerivativeFun(self, i)
		return fun

	def derivateTheta(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
			
			def cov(self, X, theta, diag=False):
				l = np.exp(2.0*self.kernel.params[0])
				lp = 2.0*np.exp(2.0*theta[0])
				sp = np.exp(2.0*theta[1])

				lpu = l + lp	
				K,R = self.kernel._compute_gausskern(lpu, X, diag=diag, retR=True)
				if i == 0:
					dlp = lp/lpu**2
				
					#K,R = self.kernel._compute_gausskern(lpu, X, diag=diag, retR=True)	
					dK = K*R*dlp
					dnorm = -np.sqrt(l)*lp/lpu**1.5
					dK = np.sqrt(l/lpu) * dK + dnorm * K
					dK = sp * dK

				elif self.i == 1:
					#dK = 2.0*sp*np.sqrt(detLpu/detL)*K
					dK = 2.0*sp*np.sqrt(l/lpu)*K
				else:
					raise ValueError('Unknown hyperparameter')
				
				return dK
			
			def ccov(self, X, Z,  thetaP, thetaQ=None, latent=False):
				l = np.exp(2.0*self.kernel.params[0])
				lp = np.exp(2.0*thetaP[0])
				sp = np.exp(thetaP[1])
				
				if latent:
					'''
					Compute cov[f_p(x), u(x')]
					'''
					lpu = l + lp
					K,R = self.kernel._compute_gausskern(lpu, X, Z, retR=True)
					if i == 0:
						dlp = lp/lpu**2
					
						dK = K*R*dlp
						dnorm = -np.sqrt(l)*lp/lpu**1.5
						dK = np.sqrt(l/lpu) * dK + dnorm * K
						dK = sp * dK
					
					elif self.i == 1:
						dK = sp*np.sqrt(l/lpu)*K
					else:
						raise ValueError('Unknown hyperparameter')
		
				else:
					'''
					Compute cov[f_p(x), f_q(x')]
					'''
					
					lq = np.exp(2*thetaQ[0])
					sq = np.exp(thetaQ[1])
						
					lpq = l+lp+lq
					#detL = np.prod(l)
					#detLpq = np.prod(lpq)
					K, R = self.kernel._compute_gausskern(lpq, X, Z, retR=True)
					if i == 0:
						dlp = lp/lpq**2
						dlq = lq/lpq**2
						
						dKp = K*R*dlp
						dKq = K*R*dlq
						
						dnorm_p = -np.sqrt(l)*lp/lpq**1.5
						dnorm_q = -np.sqrt(l)*lq/lpq**1.5
						
						dKp = np.sqrt(l/lpq) * dKp + dnorm_p * K
						dKq = np.sqrt(l/lpq) * dKq + dnorm_q * K
						
						dKp = sp * sq * dKp
						dKq = sp * sq * dKq
						
					elif self.i == 1:
						dKp = sq*sp*np.sqrt(l/lpq)*K
						dKq = sp*sq*np.sqrt(l/lpq)*K
					else:
						raise ValueError('Unknown hyperparameter')
					
					return dKp, dKq
				
				return dK			
			
			def lcov(self, X, Z=None, diag=False):
				raise NotImplementedError('latent cov has no hyperparameter theta')
			
		fun = _DerivativeFun(self, i)
		return fun

	
	def derivateX(self):
		'''
		'''
		
		class _DerivativeFunX(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def cov(self, X, theta, diag=False):
				l = np.exp(2.0*self.kernel._params[0])
				lp = 2.0*np.exp(2.0*theta[0])
				sp = np.exp(2.0*theta[1])
				
				lpu = l + lp
				
				dK = self.kernel._compute_gausskern_derivX(lpu, X, diag=diag)
				dK = sp * np.sqrt(l/lpu) * dK				
				return dK

			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				l = np.exp(2.0*self.kernel._params[0])
				lp = np.exp(2.0*thetaP[0])
				sp = np.exp(thetaP[1])
				
				if latent:
					'''
					Compute cov[f_p(x), u(x')]
					'''
					lpu = l + lp
				
					dK = self.kernel._compute_gausskern_derivX(lpu, X, Z)
					dK = sp * np.sqrt(l/lpu) * dK
		
				else:
					'''
					Compute cov[f_p(x), f_q(x')]
					'''			
					lq = np.exp(2.0*thetaQ[0])
					sq = np.exp(thetaQ[1])
					
					lpq = l+lp+lq
		
					dK = self.kernel._compute_gausskern_derivX(lpq, X, Z)
					dK = sp * sq * np.sqrt(l/lpq) * dK
				
				return dK			

			
			def lcov(self, X, Z=None, diag=False):
				l = np.exp(2*self.kernel._params[0])
				dK = self.kernel._compute_gausskern_derivX(l, X, Z, diag=diag)
				return dK
							
		fun = _DerivativeFunX(self)
		return fun
	
	def _compute_gausskern(self, l, X, Z=None, diag=False, retR=False):
		xeqz = Z==None
		
		if xeqz:	
			if diag:
				R = np.zeros(X.shape[0])
			else:
				R = distance_matrix(X,  metric='sqeuclidean') 
		else:
			R = distance_matrix(X, Z, metric='sqeuclidean')
		K = np.exp(-R/(2.0*l))
		return K if retR == False else (K,R)
	
	def _compute_gausskern_derivX(self, l, X, Z=None, diag=False):
		#P = np.diag(np.sqrt(1.0/l))
		xeqz = Z==None
		
		if xeqz:
			Z = X
		
		n,d = X.shape
		m = Z.shape[0]
		
		if xeqz and diag:
			dK = np.zeros(n*d)
		else:
			R = distance_matrix(X, Z, metric='sqeuclidean')
			K = np.exp(-R/(2.0*l))
			
			dK = np.zeros((m,n,d))
			for i in xrange(d):
				dK[:,:,i] = 1.0/l * np.add.outer(Z[:,i],-X[:,i])*K.T
		return dK

class ExpSEKernel(ExpGaussianKernel):

	'''
		todo: maybe inherit from ExpGaussianKernel
	'''

	__slots__ = ()

	def __init__(self, l, s=0.0):
		super(ExpSEKernel, self).__init__(l)
		ConvolvedKernel.__init__(self, l, 2)
		
		params = np.r_[l,s]
		self._params = params
		self._n = len(params)

	def cov(self, X, theta, diag=False):
		'''
		'''
		s = np.exp(self._params[1])
		K = super(ExpSEKernel, self).cov(X, theta, diag)
		K = s*K
		return K
	
	def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		Computes the cross covariance matrix cov[f_p(x), f_q(x')]
		Computes the cross covariance matrix cov[f_p(x), u(x')]
		'''
		s = np.exp(self._params[1])
		K = super(ExpSEKernel, self).ccov(X, Z, thetaP, thetaQ, latent)
		K = s*K
		return K
			
	def lcov(self, X, Z=None, diag=False):
		'''
		Computes the covariance matrix cov[u(x), u(x')] uof the latent process u
		'''			
		s = np.exp(2.0*self._params[1])
		K = super(ExpSEKernel, self).lcov(X, Z, diag)
		K = s*K
		return K

	def gradient_cov(self, covGrad, X, theta, diag=False):
		'''
		'''
		s = np.exp(self._params[1])
		grad_sup, gradTheta = super(ExpSEKernel, self).gradient_cov(covGrad, X, theta, diag=diag)
		
		grad = np.zeros([2])
		grad[0] = s*grad_sup
		grad[1] = np.sum(covGrad*self.cov(X, theta, diag))
		
		gradTheta = s*gradTheta
		
		return grad, gradTheta
	
	def gradient_ccov(self, covGrad, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		'''
		'''
		'''
		s = np.exp(self._params[1])
		if not latent:
			grad_sup, gradP, gradQ = super(ExpSEKernel, self).gradient_ccov(covGrad, X, Z, thetaP, thetaQ, latent)
			grad = np.zeros([2])
			grad[0] = s*grad_sup
			grad[1] = np.sum(covGrad*self.ccov(X, Z, thetaP, thetaQ, latent))
			gradP = s*gradP
			gradQ = s*gradQ
			return grad, gradP, gradQ

		else:
			grad_sup, gradP = super(ExpSEKernel, self).gradient_ccov(covGrad, X, Z, thetaP, thetaQ, latent)
			grad = np.zeros([2])
			grad[0] = s*grad_sup
			grad[1] = np.sum(covGrad*self.ccov(X, Z, thetaP, thetaQ, latent))
			gradP = s*gradP
			return grad, gradP
			
	def gradient_lcov(self, covGrad, X, Z=None, diag=False):
		s = np.exp(2.0*self._params[1])
		grad_sup = super(ExpSEKernel, self).gradient_lcov(covGrad, X, Z, diag=diag)

		grad = np.zeros([2])
		grad[0] = s*grad_sup
		grad[1] = 2.0*np.sum(covGrad*self.lcov(X, Z, diag))

		return grad
	
		
	def derivate(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
			
			def cov(self, X, theta, diag=False):
				kernel = self.kernel
				i = self.i 
				s = np.exp(self.kernel._params[1])
				
				if i==0:
					dkernel = super(ExpSEKernel, kernel).derivate(i)
					dK = dkernel.cov(X, theta, diag)
					dK = s*dK
				elif i==1:
					dK = self.kernel.cov(X, theta, diag)
				else:
					raise ValueError("unknown hyperparameter")
				return dK
			
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				kernel = self.kernel
				i = self.i 
				s = np.exp(self.kernel._params[1])
				
				if i==0:
					dkernel = super(ExpSEKernel, kernel).derivate(i)
					dK = dkernel.ccov(X, Z, thetaP, thetaQ, latent)
					dK = s*dK
				elif i==1:
					dK = self.kernel.ccov(X, Z, thetaP, thetaQ, latent)
				else:
					raise ValueError("unknown hyperparameter")
				return dK
						
			
			def lcov(self, X, Z=None, diag=False):
				kernel = self.kernel
				i = self.i 
				s = np.exp(2.0*self.kernel._params[1])
				
				if i==0:
					dkernel = super(ExpSEKernel, kernel).derivate(i)
					dK = dkernel.lcov(X, Z, diag)
					dK = s*dK
				elif i==1:
					dK = self.kernel.lcov(X, Z, diag)
					dK = 2.0*dK
				else:
					raise ValueError("unknown hyperparameter")
				return dK

		if i >= self.nparams:	
			raise TypeError('Unknown hyperparameter')
								
		fun = _DerivativeFun(self, i)
		return fun

	def derivateTheta(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.dkernel = super(ExpSEKernel, kernel).derivateTheta(i)
				self.i = i
			
			def cov(self, X, theta, diag=False):
				s = np.exp(self.kernel._params[1])
				dK = self.dkernel.cov(X, theta, diag)
				dK = s*dK
				return dK
						
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				s = np.exp(self.kernel._params[1])
				
				if not latent:
					dKp, dKq = self.dkernel.ccov(X, Z, thetaP, thetaQ, latent)
					dKp = s*dKp
					dKq = s*dKq
					return dKp, dKq
				else:
					dK = self.dkernel.ccov(X, Z, thetaP, thetaQ, latent)
					dK = s*dK
					return dK
							
			def lcov(self, X, diag=False):
				raise NotImplementedError('latent cov has no hyperparameter theta')
			
		fun = _DerivativeFun(self, i)
		return fun

	
	def derivateX(self):
		'''
		'''
		
		class _DerivativeFunX(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				self.dkernel = super(ExpSEKernel, kernel).derivateX()
				
			def cov(self, X, theta, diag=False):
				s = np.exp(self.kernel._params[1])
				dK = self.dkernel.cov(X, theta, diag)
				dK = s*dK
				return dK
			
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				s = np.exp(self.kernel._params[1])
				dK = self.dkernel.ccov(X, Z, thetaP, thetaQ, latent)
				dK = s*dK
				return dK
			
			def lcov(self, X, Z=None, diag=False):
				s = np.exp(self.kernel._params[1])
				dK = self.dkernel.lcov(X, Z, diag)
				dK = s*dK
				return dK
				
							
		fun = _DerivativeFunX(self)
		return fun

		
class ExpARDGaussianKernel(ConvolvedKernel):

	__slots__ = ('_d')

	def __init__(self, l):
		'''
		'''
		l = np.asarray(l).ravel()
		d = len(l)
		ConvolvedKernel.__init__(self, l, d+1)
		self._d = d


	def cov(self, X, theta, diag=False):
		'''
		'''
		d = self._d
		l = np.exp(2.0*self.params[0:d])
		lp = 2.0*np.exp(2.0*theta[0:d])
		sp = np.exp(2.0*theta[d])
		
		lpu = l + lp
		detL = np.prod(l)
		detLpu = np.prod(lpu)
		K = self._compute_gausskern(lpu, X, diag=diag)
		#K = sp * np.sqrt(detLpu/detL) * K
		K = sp * np.sqrt(detL/detLpu) * K
		
		return K
	
	def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		Computes the cross covariance matrix cov[f_p(x), f_q(x')]
		Computes the cross covariance matrix cov[f_p(x), u(x')]
		'''
		d = self._d
		l = np.exp(2.0*self.params[0:d])
		lp = np.exp(2.0*thetaP[0:d])
		sp = np.exp(thetaP[d])
		
		if latent:
			'''
			Compute cov[f_p(x), u(x')]
			'''
			lpu = l + lp
			detL = np.prod(l)
			detLpu = np.prod(lpu)
		
			K = self._compute_gausskern(lpu, X, Z)
			#K = sp * np.sqrt(detLpu/detL) * K
			K = sp * np.sqrt(detL/detLpu) * K

		else:
			'''
			Compute cov[f_p(x), f_q(x')]
			'''			
			lq = np.exp(2.0*thetaQ[0:d])
			sq = np.exp(thetaQ[d])
			
			lpq = l+lp+lq
			detL = np.prod(l)
			detLpq = np.prod(lpq)

			K = self._compute_gausskern(lpq, X, Z)
			#K = sp * sq * np.sqrt(detLpq/detL) * K
			K = sp * sq * np.sqrt(detL/detLpq) * K
		
		return K			
		
	def lcov(self, X, Z=None, diag=False):
		'''
		Computes the covariance matrix cov[u(x), u(x')] uof the latent process u
		'''			
		d = self._d
		l = np.exp(2.0*self.params[0:d])
		K = self._compute_gausskern(l, X, Z, diag=diag)
		return K

	def gradient_cov(self, covGrad, X, theta, diag=False):
		'''
		'''
		d = self._d
		l = np.exp(2*self.params[0:d])
		lp = 2*np.exp(2.0*theta[0:d])
		sp = np.exp(2.0*theta[d])
		
		lpu = l + lp
		detL = np.prod(l)
		detLpu = np.prod(lpu)
		
		K = self._compute_gausskern(lpu, X, diag=diag)
		
		grad = np.empty(d)
		gradTheta = np.empty(d+1)
		for i in xrange(d):
			detLpui = l[i]*detLpu/lpu[i]
			detLpi = lp[i]*detLpu/lpu[i]
			dl = np.sqrt(l[i])/lpu[i]
			dlp = np.sqrt(lp[i])/lpu[i]
		
			if diag:
				Kprime = 0
			else:
				Kprime = distance_matrix(X[:,i,np.newaxis]*dl, metric='sqeuclidean')
				Kprime_p = distance_matrix(X[:,i,np.newaxis]*dlp, metric='sqeuclidean')
			
			dK = K*Kprime
			dKp = K*Kprime_p
			dnorm = np.sqrt(detL)*(1.0/np.sqrt(detLpu)-detLpui/detLpu**(1.5))
			dnorm_p = -detLpi*np.sqrt(detL)/detLpu**1.5
			dK = np.sqrt(detL/detLpu) * dK + dnorm * K
			dK = sp * dK
			dKp = np.sqrt(detL/detLpu) * dKp + dnorm_p  * K
			dKp = sp*dKp
			grad[i] = np.sum(covGrad*dK)
			gradTheta[i] = np.sum(covGrad*dKp)
			
		gradTheta[d] = 2.0*sp*np.sqrt(detL/detLpu)*np.sum(covGrad*K)
		return grad, gradTheta
	
	def gradient_ccov(self, covGrad, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		'''
		'''
		'''
		d = self._d
		l = np.exp(2.0*self.params[0:d])
		lp = np.exp(2.0*thetaP[0:d])
		sp = np.exp(thetaP[d])
		
		if latent:
			'''
			Compute cov[f_p(x), u(x')]
			'''
			lpu = l + lp
			detL = np.prod(l)
			detLpu = np.prod(lpu)

			K = self._compute_gausskern(lpu, X, Z)
									
			grad = np.empty(d)
			gradTheta = np.empty(d+1)
			for i in xrange(d):			
				detLpui = l[i]*detLpu/lpu[i]
				detLpi = lp[i]*detLpu/lpu[i]
				dl = np.sqrt(l[i])/lpu[i]
				dlp = np.sqrt(lp[i])/lpu[i]

				Kprime = distance_matrix(X[:,i,np.newaxis]*dl, Z[:,i,np.newaxis]*dl, metric='sqeuclidean')
				Kprime_p = distance_matrix(X[:,i,np.newaxis]*dlp, Z[:,i,np.newaxis]*dlp, metric='sqeuclidean')
				
				dK = K*Kprime
				dKp = K*Kprime_p
				#dnorm = detLpui/np.sqrt(detL*detLpu) - np.sqrt(detLpu/detL) 
				#dK = np.sqrt(detLpu/detL) * dK + dnorm * K
				dnorm = np.sqrt(detL)*(1/np.sqrt(detLpu)-detLpui/detLpu**(1.5))
				dnorm_p = -detLpi*np.sqrt(detL)/detLpu**1.5
				dK = np.sqrt(detL/detLpu) * dK + dnorm * K
				dK = sp * dK
				dKp = np.sqrt(detL/detLpu) * dKp + dnorm_p * K
				dKp = sp * dKp
				
				grad[i] = np.sum(covGrad*dK)
				gradTheta[i] = np.sum(covGrad*dKp)
				
			gradTheta[d] = sp*np.sqrt(detL/detLpu)*np.sum(covGrad*K)
			return grad, gradTheta


		else:
			'''
			Compute cov[f_p(x), f_q(x')]
			'''
			lq = np.exp(2*thetaQ[0:d])
			sq = np.exp(thetaQ[d])
			
			lpq = l+lp+lq
			detL = np.prod(l)
			detLpq = np.prod(lpq)
			
			K = self._compute_gausskern(lpq, X, Z)
			
			grad = np.empty(d)
			gradTheta_p = np.empty(d+1)
			gradTheta_q = np.empty(d+1)
			for i in xrange(d):
				detLpqi = l[i]*detLpq/lpq[i]
				detLpi = lp[i]*detLpq/lpq[i]
				detLqi = lq[i]*detLpq/lpq[i]
				dl = np.sqrt(l[i])/lpq[i]
				dlp = np.sqrt(lp[i])/lpq[i]
				dlq = np.sqrt(lq[i])/lpq[i]
			
				Kprime = distance_matrix(X[:,i,np.newaxis]*dl, Z[:,i,np.newaxis]*dl, metric='sqeuclidean')
				Kprime_p = distance_matrix(X[:,i,np.newaxis]*dlp, Z[:,i,np.newaxis]*dlp, metric='sqeuclidean')
				Kprime_q = distance_matrix(X[:,i,np.newaxis]*dlq, Z[:,i,np.newaxis]*dlq, metric='sqeuclidean')
				dK = K*Kprime
				dKp = K*Kprime_p
				dKq = K*Kprime_q
				#dnorm = detLpqi/np.sqrt(detL*detLpq) - np.sqrt(detLpq/detL)
				#dK = np.sqrt(detLpq/detL) * dK + dnorm * K
				dnorm = np.sqrt(detL)*(1/np.sqrt(detLpq)-detLpqi/detLpq**(1.5))
				dnorm_p = -detLpi*np.sqrt(detL)/detLpq**1.5
				dnorm_q = -detLqi*np.sqrt(detL)/detLpq**1.5
				
				dK = np.sqrt(detL/detLpq) * dK + dnorm * K
				dK = sp * sq * dK
				dKp = np.sqrt(detL/detLpq) * dKp + dnorm_p * K
				dKp = sp * sq * dKp
				#dKq = np.sqrt(detLpq/detL) * dKq + dnorm_q * K
				dKq = np.sqrt(detL/detLpq) * dKq + dnorm_q * K
				dKq = sp * sq * dKq
				
				grad[i] = np.sum(covGrad*dK)
				gradTheta_p[i] = np.sum(covGrad*dKp)
				gradTheta_q[i] = np.sum(covGrad*dKq)

			gradTheta_p[d] = sp*sq*np.sqrt(detL/detLpq)*np.sum(covGrad*K)
			gradTheta_q[d] = sp*sq*np.sqrt(detL/detLpq)*np.sum(covGrad*K)
			return grad, gradTheta_p, gradTheta_q
	
	def gradient_lcov(self, covGrad, X, Z=None, diag=False):
		xeqz = Z == None
		
		d = self._d
		l = np.exp(2.0*self.params[0:d])
		K = self._compute_gausskern(l, X, Z, diag=diag)
		
		grad = np.empty(d)
		#gradient of the length scales l
		for i in xrange(d):
			if xeqz:
				if diag:
					Kprime = 0.0
				else:
					Kprime = distance_matrix(X[:,i,np.newaxis]/l[i], metric='sqeuclidean')
			else:
				Kprime = distance_matrix(X[:,i,np.newaxis]/l[i], 
										 Z[:,i,np.newaxis]/l[i], metric='sqeuclidean')
			dK = K*Kprime
			grad[i] = np.sum(covGrad*dK)
		
		return grad


	def derivate(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
			
			def cov(self, X, theta, diag=False):
				i = self.i
				d = self.kernel._d
				l = np.exp(2*self.kernel.params[0:d])
				lp = 2*np.exp(2.0*theta[0:d])
				sp = np.exp(2.0*theta[d])
				
				lpu = l + lp
				detL = np.prod(l)
				detLpu = np.prod(lpu)
				detLpui = l[i]*detLpu/lpu[i]
				dl = np.sqrt(l[i])/lpu[i]
				
				K = self.kernel._compute_gausskern(lpu, X, diag=diag)
				if diag:
					Kprime = 0
				else:
					Kprime = distance_matrix(X[:,i,np.newaxis]*dl, metric='sqeuclidean')
					
				dK = K*Kprime
				#dnorm = detLpui/np.sqrt(detL*detLpu) - np.sqrt(detLpu/detL) 
				#dK = np.sqrt(detLpu/detL) * dK + dnorm * K
				dnorm = np.sqrt(detL)*(1.0/np.sqrt(detLpu)-detLpui/detLpu**(1.5))
				dK = np.sqrt(detL/detLpu) * dK + dnorm * K
				dK = sp * dK
				
				return dK
			
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				i = self.i
				d = self.kernel._d
				l = np.exp(2.0*self.kernel.params[0:d])
				lp = np.exp(2.0*thetaP[0:d])
				sp = np.exp(thetaP[d])
				
				if latent:
					'''
					Compute cov[f_p(x), u(x')]
					'''
					lpu = l + lp
					detL = np.prod(l)
					detLpu = np.prod(lpu)
					detLpui = l[i]*detLpu/lpu[i]
					dl = np.sqrt(l[i])/lpu[i]
					
					K = self.kernel._compute_gausskern(lpu, X, Z)
					Kprime = distance_matrix(X[:,i,np.newaxis]*dl, Z[:,i,np.newaxis]*dl, metric='sqeuclidean')
					dK = K*Kprime
					#dnorm = detLpui/np.sqrt(detL*detLpu) - np.sqrt(detLpu/detL) 
					#dK = np.sqrt(detLpu/detL) * dK + dnorm * K
					dnorm = np.sqrt(detL)*(1/np.sqrt(detLpu)-detLpui/detLpu**(1.5))
					dK = np.sqrt(detL/detLpu) * dK + dnorm * K
					dK = sp * dK
		
				else:
					'''
					Compute cov[f_p(x), f_q(x')]
					'''
					lq = np.exp(2*thetaQ[0:d])
					sq = np.exp(thetaQ[d])
					
					lpq = l+lp+lq
					detL = np.prod(l)
					detLpq = np.prod(lpq)
					detLpqi = l[i]*detLpq/lpq[i]
					dl = np.sqrt(l[i])/lpq[i]
					
					K = self.kernel._compute_gausskern(lpq, X, Z)
					Kprime = distance_matrix(X[:,i,np.newaxis]*dl, Z[:,i,np.newaxis]*dl, metric='sqeuclidean')
					dK = K*Kprime
					#dnorm = detLpqi/np.sqrt(detL*detLpq) - np.sqrt(detLpq/detL)
					#dK = np.sqrt(detLpq/detL) * dK + dnorm * K
					dnorm = np.sqrt(detL)*(1/np.sqrt(detLpq)-detLpqi/detLpq**(1.5)) 
					dK = np.sqrt(detL/detLpq) * dK + dnorm * K
					dK = sp * sq * dK
			
				return dK			
			
			def lcov(self, X, Z=None, diag=False):
				i = self.i
				d = self.kernel._d
				xeqz = Z == None
				if i < d:	
					l = np.exp(2*self.kernel.params[0:d])
					K = self.kernel._compute_gausskern(l, X, Z, diag=diag)
					if xeqz:
						if diag:
							Kprime = 0.0
						else:
							Kprime = distance_matrix(X[:,i,np.newaxis]/np.sqrt(l[i]), metric='sqeuclidean')
					else:
						Kprime = distance_matrix(X[:,i,np.newaxis]/np.sqrt(l[i]), 
												 Z[:,i,np.newaxis]/np.sqrt(l[i]), metric='sqeuclidean')
					dK = K*Kprime
				else:
					raise TypeError('Unknown hyperparameter')
				return dK

		if i >= self._d:	
			raise TypeError('Unknown hyperparameter')
								
		fun = _DerivativeFun(self, i)
		return fun

	def derivateTheta(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
			
			def cov(self, X, theta, diag=False):
				d = self.kernel._d
				l = np.exp(2.0*self.kernel.params[0:d])
				lp = 2.0*np.exp(2.0*theta[0:d])
				sp = np.exp(2.0*theta[d])
				
				
				lpu = l + lp
				detL = np.prod(l)
				detLpu = np.prod(lpu)
				
				K = self.kernel._compute_gausskern(lpu, X, diag=diag)
				if i < d:
					detLpui = lp[i]*detLpu/lpu[i]
					dlp = np.sqrt(lp[i])/lpu[i]
					
					if diag:
						Kprime = 0.0
					else:
						Kprime = distance_matrix(X[:,i,np.newaxis]*dlp, metric='sqeuclidean')
						
					dK = K*Kprime
					#dnorm = detLpui/np.sqrt(detL*detLpu) 
					#dK = np.sqrt(detLpu/detL) * dK + dnorm  * K
					dnorm = -detLpui*np.sqrt(detL)/detLpu**1.5
					dK = np.sqrt(detL/detLpu) * dK + dnorm  * K
					dK = sp*dK
				elif self.i == d:
					#dK = 2.0*sp*np.sqrt(detLpu/detL)*K
					dK = 2.0*sp*np.sqrt(detL/detLpu)*K
				else:
					raise ValueError('Unknown hyperparameter')
				
				return dK
			
			def ccov(self, X, Z,  thetaP, thetaQ=None, latent=False):
				d = self.kernel._d
				l = np.exp(2.0*self.kernel.params[0:d])
				lp = np.exp(2.0*thetaP[0:d])
				sp = np.exp(thetaP[d])
				
				if latent:
					'''
					Compute cov[f_p(x), u(x')]
					'''
					lpu = l + lp
					detL = np.prod(l)
					detLpu = np.prod(lpu)
					
					K = self.kernel._compute_gausskern(lpu, X, Z)
					if i < d:
						detLpui = lp[i]*detLpu/lpu[i]
						dlp = np.sqrt(lp[i])/lpu[i]	
						
						Kprime = distance_matrix(X[:,i,np.newaxis]*dlp, Z[:,i,np.newaxis]*dlp, metric='sqeuclidean')
						dK = K*Kprime
						#dnorm = detLpui/np.sqrt(detL*detLpu) 
						#dK = np.sqrt(detLpu/detL) * dK + dnorm * K
						dnorm = -detLpui*np.sqrt(detL)/detLpu**1.5
						dK = np.sqrt(detL/detLpu) * dK + dnorm * K
						dK = sp * dK
					elif self.i == d:
						dK = sp*np.sqrt(detL/detLpu)*K
					else:
						raise ValueError('Unknown hyperparameter')
		
				else:
					'''
					Compute cov[f_p(x), f_q(x')]
					'''
					
					lq = np.exp(2.0*thetaQ[0:d])
					sq = np.exp(thetaQ[d])
						
					lpq = l+lp+lq
					detL = np.prod(l)
					detLpq = np.prod(lpq)
					K = self.kernel._compute_gausskern(lpq, X, Z)
					if i < d:
						detLpi = lp[i]*detLpq/lpq[i]
						detLqi = lq[i]*detLpq/lpq[i]
						dlp = np.sqrt(lp[i])/lpq[i]
						dlq = np.sqrt(lq[i])/lpq[i]
						
						Kprime_p = distance_matrix(X[:,i,np.newaxis]*dlp, Z[:,i,np.newaxis]*dlp, metric='sqeuclidean')
						Kprime_q = distance_matrix(X[:,i,np.newaxis]*dlq, Z[:,i,np.newaxis]*dlq, metric='sqeuclidean')
						dKp = K*Kprime_p
						dKq = K*Kprime_q
						#dnorm_p = detLpi/np.sqrt(detL*detLpq) 
						#dnorm_q = detLqi/np.sqrt(detL*detLpq)
						dnorm_p = -detLpi*np.sqrt(detL)/detLpq**1.5
						dnorm_q = -detLqi*np.sqrt(detL)/detLpq**1.5
						#dKp = np.sqrt(detLpq/detL) * dKp + dnorm_p * K
						dKp = np.sqrt(detL/detLpq) * dKp + dnorm_p * K
						dKp = sp * sq * dKp
						#dKq = np.sqrt(detLpq/detL) * dKq + dnorm_q * K
						dKq = np.sqrt(detL/detLpq) * dKq + dnorm_q * K
						dKq = sp * sq * dKq
					elif self.i == d:
						dKp = sq*sp*np.sqrt(detL/detLpq)*K
						dKq = sp*sq*np.sqrt(detL/detLpq)*K
					else:
						raise ValueError('Unknown hyperparameter')
					
					return dKp, dKq
				
				return dK			
			
			def lcov(self, X, Z=None, diag=False):
				raise NotImplementedError('latent cov has no hyperparameter theta')
			
		fun = _DerivativeFun(self, i)
		return fun

	
	def derivateX(self):
		'''
		'''
		
		class _DerivativeFunX(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				
			def cov(self, X, theta, diag=False):
				d = self.kernel._d
				l = np.exp(2.0*self.kernel._params[0:d])
				lp = 2.0*np.exp(2.0*theta[0:d])
				sp = np.exp(2.0*theta[d])
				
				lpu = l + lp
				detL = np.prod(l)
				detLpu = np.prod(lpu)
				
				dK = self.kernel._compute_gausskern_derivX(lpu, X, diag=diag)
				dK = sp * np.sqrt(detL/detLpu) * dK				
				return dK

			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				d = self.kernel._d
				l = np.exp(2.0*self.kernel._params[0:d])
				lp = np.exp(2.0*thetaP[0:d])
				sp = np.exp(thetaP[d])
				
				if latent:
					'''
					Compute cov[f_p(x), u(x')]
					'''
					lpu = l + lp
					detL = np.prod(l)
					detLpu = np.prod(lpu)
				
					dK = self.kernel._compute_gausskern_derivX(lpu, X, Z)
					dK = sp * np.sqrt(detL/detLpu) * dK
		
				else:
					'''
					Compute cov[f_p(x), f_q(x')]
					'''			
					lq = np.exp(2*thetaQ[0:d])
					sq = np.exp(thetaQ[d])
					
					lpq = l+lp+lq
					detL = np.prod(l)
					detLpq = np.prod(lpq)
		
					dK = self.kernel._compute_gausskern_derivX(lpq, X, Z)
					dK = sp * sq * np.sqrt(detL/detLpq) * dK
				
				return dK			

			
			def lcov(self, X, Z=None, diag=False):
				d = self.kernel._d
				l = np.exp(2.0*self.kernel._params[0:d])
				dK = self.kernel._compute_gausskern_derivX(l, X, Z, diag=diag)
				return dK
							
		fun = _DerivativeFunX(self)
		return fun
	
	def _compute_gausskern(self, l, X, Z=None, diag=False, retR=False):
		P = np.diag(np.sqrt(1.0/l))
		xeqz = Z==None
		
		if xeqz:	
			if diag:
				K = np.zeros(X.shape[0])
			else:
				K = distance_matrix(np.dot(P, X.T).T,  metric='sqeuclidean') 
		else:
			K = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, metric='sqeuclidean')
		K = np.exp(-K/2)
		return K

	def _compute_gausskern_old(self, l, X, Z=None, diag=False, retR=False):
		xeqz = Z==None
		
		if xeqz:	
			if diag:
				R = np.zeros(X.shape[0])
			else:
				R = distance_matrix(X,  metric='sqeuclidean') 
		else:
			R = distance_matrix(X, Z, metric='sqeuclidean')
		K = np.exp(-R/(2.0*l))
		return K if retR == False else (K,R)

	
	def _compute_gausskern_derivX(self, l, X, Z=None, diag=False):
		P = np.diag(np.sqrt(1.0/l))
		xeqz = Z==None
		
		if xeqz:
			Z = X
		
		n,d = X.shape
		m = Z.shape[0]
		
		if xeqz and diag:
			dK = np.zeros(n*d)
		else:
			R = distance_matrix(np.dot(P, X.T).T, np.dot(P, Z.T).T, metric='sqeuclidean')
			K = np.exp(-R/2)
			
			dK = np.zeros((m,n,d))
			for i in xrange(d):
				dK[:,:,i] = 1/l[i] * np.add.outer(Z[:,i],-X[:,i])*K.T
		return dK
	
class ExpARDSEKernel(ExpARDGaussianKernel):

	'''
		todo: maybe inherit from ExpGaussianKernel
	'''

	__slots__ = ()

	def __init__(self, l, s=0.0):
		l = np.asarray(l).ravel()
		d = len(l)
		super(ExpARDSEKernel, self).__init__(l)
		ConvolvedKernel.__init__(self, l, d+1)
		
		params = np.r_[l,s]
		self._params = params
		self._n = len(params)

	def cov(self, X, theta, diag=False):
		'''
		'''
		d = self._d
		s = np.exp(self._params[d])
		K = super(ExpARDSEKernel, self).cov(X, theta, diag)
		K = s*K
		return K
	
	def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		Computes the cross covariance matrix cov[f_p(x), f_q(x')]
		Computes the cross covariance matrix cov[f_p(x), u(x')]
		'''
		d = self._d
		s = np.exp(self._params[d])
		K = super(ExpARDSEKernel, self).ccov(X, Z, thetaP, thetaQ, latent)
		K = s*K
		return K
			
	def lcov(self, X, Z=None, diag=False):
		'''
		Computes the covariance matrix cov[u(x), u(x')] uof the latent process u
		'''			
		d = self._d 
		s = np.exp(2.0*self._params[d])
		K = super(ExpARDSEKernel, self).lcov(X, Z, diag)
		K = s*K
		return K

	def gradient_cov(self, covGrad, X, theta, diag=False):
		'''
		'''
		d = self._d
		s = np.exp(self._params[d])
		grad_sup, gradTheta = super(ExpARDSEKernel, kernel).gradient_cov(covGrad, X, theta, diag=diag)
		
		grad = np.zeros([d+1])
		grad[0:d] = s*grad_sup
		grad[d] = np.sum(covGrad*self.cov(X, theta, diag))
		
		gradTheta = s*gradTheta
		
		return grad, gradTheta
	
	def gradient_ccov(self, covGrad, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		'''
		'''
		'''
		d = self._d
		s = np.exp(self._params[d])
		if not latent:
			grad_sup, gradP, gradQ = super(ExpARDSEKernel, kernel).gradient_ccov(covGrad, X, Z, thetaP, thetaQ, latent)
			grad = np.zeros([d+1])
			grad[0:d] = s*grad_sup
			grad[d] = np.sum(covGrad*self.ccov(X, Z, thetaP, thetaQ, latent))
			gradP = s*gradP
			gradQ = s*gradQ
			return grad, gradP, gradQ

		else:
			grad_sup, gradP = super(ExpARDSEKernel, kernel).gradient_ccov(covGrad, X, Z, thetaP, thetaQ, latent)
			grad = np.zeros([d+1])
			grad[0:d] = s*grad_sup
			grad[d] = np.sum(covGrad*self.ccov(X, Z, thetaP, thetaQ, latent))
			gradP = s*gradP
			return grad, gradP
			
	def gradient_lcov(self, covGrad, X, Z=None, diag=False):
		d = self._d
		s = np.exp(2.0*self._params[d])
		grad_sup = super(ExpARDSEKernel, kernel).gradient_lcov(covGrad, X, Z, diag=diag)

		grad = np.zeros([d+1])
		grad[0:d] = s*grad_sup
		grad[d] = 2.0*np.sum(covGrad*self.lcov(X, Z, diag))

		return grad			
		
	def derivate(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.i = i
			
			def cov(self, X, theta, diag=False):
				kernel = self.kernel
				i = self.i
				d = self.kernel._d 
				s = np.exp(self.kernel._params[d])
				
				if i<d:
					dkernel = super(ExpARDSEKernel, kernel).derivate(i)
					dK = dkernel.cov(X, theta, diag)
					dK = s*dK
				elif i==d:
					dK = self.kernel.cov(X, theta, diag)
				else:
					raise ValueError("unknown hyperparameter")
				return dK
			
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				kernel = self.kernel
				i = self.i
				d = self.kernel._d 
				s = np.exp(self.kernel._params[d])
				
				if i<d:
					dkernel = super(ExpARDSEKernel, kernel).derivate(i)
					dK = dkernel.ccov(X, Z, thetaP, thetaQ, latent)
					dK = s*dK
				elif i==d:
					dK = self.kernel.ccov(X, Z, thetaP, thetaQ, latent)
				else:
					raise ValueError("unknown hyperparameter")
				return dK
						
			
			def lcov(self, X, Z=None, diag=False):
				kernel = self.kernel
				i = self.i
				d = self.kernel._d 
				s = np.exp(2*self.kernel._params[d])
				
				if i<d:
					dkernel = super(ExpARDSEKernel, kernel).derivate(i)
					dK = dkernel.lcov(X, Z, diag)
					dK = s*dK
				elif i==d:
					dK = self.kernel.lcov(X, Z, diag)
					dK = 2*dK
				else:
					raise ValueError("unknown hyperparameter")
				return dK

		if i >= self.nparams:	
			raise TypeError('Unknown hyperparameter')
								
		fun = _DerivativeFun(self, i)
		return fun

	def derivateTheta(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, i):
				self.kernel = kernel
				self.dkernel = super(ExpARDSEKernel, kernel).derivateTheta(i)
				self.i = i
			
			def cov(self, X, theta, diag=False):
				d = self.kernel._d
				s = np.exp(self.kernel._params[d])
				
				dK = self.dkernel.cov(X, theta, diag)
				dK = s*dK
				return dK
						
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				d = self.kernel._d
				s = np.exp(self.kernel._params[d])
				
				if not latent:
					dKp, dKq = self.dkernel.ccov(X, Z, thetaP, thetaQ, latent)
					dKp = s*dKp
					dKq = s*dKq
					return dKp, dKq
				else:
					dK = self.dkernel.ccov(X, Z, thetaP, thetaQ, latent)
					dK = s*dK
					return dK
							
			def lcov(self, X, diag=False):
				raise NotImplementedError('latent cov has no hyperparameter theta')
			
		fun = _DerivativeFun(self, i)
		return fun

	
	def derivateX(self):
		'''
		'''
		
		class _DerivativeFunX(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernel = kernel
				self.dkernel = super(ExpARDSEKernel, kernel).derivateX()
				
			def cov(self, X, theta, diag=False):
				d = self.kernel._d
				s = np.exp(self.kernel._params[d])
				dK = self.dkernel.cov(X, theta, diag)
				dK = s*dK
				return dK
			
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				d = self.kernel._d
				s = np.exp(self.kernel._params[d])
				dK = self.dkernel.ccov(X, Z, thetaP, thetaQ, latent)
				dK = s*dK
				return dK
			
			def lcov(self, X, Z=None, diag=False):
				d = self.kernel._d
				s = np.exp(self.kernel._params[d])
				dK = self.dkernel.lcov(X, Z, diag)
				dK = s*dK
				return dK
				
							
		fun = _DerivativeFunX(self)
		return fun


class CompoundKernel(ConvolvedKernel):
	
	__slots__ = ('_kernels',
				 '_q',
				 '_theta_idx')
	
	def __init__(self, kernels):
		q = len(kernels)
		array = np.empty(q, dtype='object')
		#array = ()
		theta_idx = np.zeros(q+1, dtype='int')
		for i in xrange(q):
			array[i] = kernels[i].params
			theta_idx[i+1] = theta_idx[i]+kernels[i].ntheta
		params = CompoundKernel.MixedArray(array)
		
		self._params = params
		self._n = len(params)
		self._q = q
		self._kernels = kernels
		self._theta_idx = theta_idx
		self._ntheta = theta_idx[q]

	def cov(self, X, theta, diag=False):
		'''
		'''
		q = self._q
		kernels = self._kernels
		theta_idx = self._theta_idx
		
		n = len(X)
		K = np.zeros((n,n)) if diag == False else np.zeros(n)
		for i in xrange(q):
			start = theta_idx[i]
			end = theta_idx[i+1]
			theta_i = theta[start:end]
			K += kernels[i].cov(X, theta_i, diag)
		return K

	def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		Computes the cross covariance matrix cov[f_p(x), f_q(x)]
		Computes the cross covariance matrix cov[f_p(x), f_q(x)]
		'''
		q = self._q
		kernels = self._kernels
		theta_idx = self._theta_idx
				
		n = len(X)
		m = len(Z)
		K = np.zeros((n,m))
		for i in xrange(q):
			start = theta_idx[i]
			end = theta_idx[i+1]
			thetaPi = thetaP[start:end]
			if latent:
				K += kernels[i].ccov(X, Z, thetaPi, latent=latent)
			else:
				thetaQi = thetaQ[start:end]
				K += kernels[i].ccov(X, Z, thetaPi, thetaQi, latent=latent)
		return K

	def lcov(self, X, Z=None, diag=False):
		'''
		Computes the covariance matrix cov[u(x), u(x')] uof the latent process u
		'''
		q = self._q
		kernels = self._kernels
		xeqz = Z == None
		n = len(X)
		if xeqz:
			K = np.zeros((n,n)) if diag == False else np.zeros(n)
		else:
			m = len(Z)
			K = np.zeros((n,m))
		for i in xrange(q):
			K += kernels[i].lcov(X, Z, diag)
		return K
	
	def gradient_cov(self, covGrad, X, theta, diag=False):
		'''
		'''
		q = self._q
		kernels = self._kernels
		theta_idx = self._theta_idx
		ntheta = self.ntheta
		nparams = self.nparams
		
		grad = np.zeros(nparams)
		gradTheta = np.zeros(ntheta)
		pos = 0
		for i in xrange(q):
			start = theta_idx[i]
			end = theta_idx[i+1]
			offset = kernels[i].nparams
			grad[pos:pos+offset], gradTheta[start:end] = kernels[i].gradient_cov(covGrad, X, theta, diag=diag)
			pos += offset
				
		return grad, gradTheta
	
	def gradient_ccov(self, covGrad, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		'''
		'''
		'''
		q = self._q
		kernels = self._kernels
		theta_idx = self._theta_idx
		ntheta = self.ntheta
		nparams = self.nparams
		
		grad = np.zeros(nparams)
		
		if not latent:
			gradP = np.zeros(ntheta)
			gradQ = np.zeros(ntheta)
			
			pos = 0
			for i in xrange(q):
				start = theta_idx[i]
				end = theta_idx[i+1]
				offset = kernels[i].nparams
				grad[pos:pos+offset], gradP[start:end], gradQ[start:end] = kernels[i].gradient_ccov(covGrad, X, Z, thetaP, thetaQ, latent)
				pos += offset
				
			return grad, gradP, gradQ

		else:
			gradP = np.zeros(ntheta)
			
			pos = 0
			for i in xrange(q):
				start = theta_idx[i]
				end = theta_idx[i+1]
				offset = kernels[i].nparams
				grad[pos:pos+offset], gradP[start:end] = kernels[i].gradient_ccov(covGrad, X, Z, thetaP, thetaQ, latent)
				pos += offset
				
			return grad, gradP
			
	def gradient_lcov(self, covGrad, X, Z=None, diag=False):
		q = self._q
		kernels = self._kernels
		nparams = self.nparams
		
		grad = np.zeros(nparams)
		pos = 0
		for i in xrange(q):
			offset = kernels[i].nparams
			grad[pos:pos+offset] = kernels[i].gradient_lcov(covGrad, X, Z, diag=diag)
			pos += offset
				
		return grad
	
	def derivate(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		'''
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, dkernel, start_idx, end_idx):
				self.dkernel = dkernel
				self.start_idx = start_idx
				self.end_idx = end_idx
			
			def cov(self, X, theta, diag=False):
				dkernel = self.dkernel
				start_idx = self.start_idx
				end_idx = self.end_idx
				dK = dkernel.cov(X, theta[start_idx:end_idx], diag)
				return dK
						
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				dkernel = self.dkernel
				start_idx = self.start_idx
				end_idx = self.end_idx
				if thetaQ == None:
					dK = dkernel.ccov(X, Z, thetaP[start_idx:end_idx], latent=latent)
					return dK
				else:
					dK = dkernel.ccov(X, Z, thetaP[start_idx:end_idx], thetaQ[start_idx:end_idx], latent=latent)
					return dK
							
			def lcov(self, X, Z=None, diag=False):
				dKernel = self.dkernel
				dK = dKernel.lcov(X, Z, diag)
				return dK 

		
		u, i, k = self._lookup_kernel_and_param(i)
		dKernel = u.derivate(i)
		return _DerivativeFun(dKernel, self._theta_idx[k], self._theta_idx[k+1])	

	def derivateTheta(self, i):
		'''
		Returns the derivative of the kernel with respect to the i'th parameter.
		@todo: - eventually, use lambda function
		'''
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, dkernel, start_idx, end_idx):
				self.dkernel = dkernel
				self.start_idx = start_idx
				self.end_idx = end_idx
			
			def cov(self, X, theta, diag=False):
				dkernel = self.dkernel
				start_idx = self.start_idx
				end_idx = self.end_idx
				dK = dkernel.cov(X, theta[start_idx:end_idx], diag)
				return dK
						
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				dkernel = self.dkernel
				start_idx = self.start_idx
				end_idx = self.end_idx
				if thetaQ == None:
					dK = dkernel.ccov(X, Z, thetaP[start_idx:end_idx], latent=latent)
					return dK
				else:
					dKp, dKq = dkernel.ccov(X, Z, thetaP[start_idx:end_idx], thetaQ[start_idx:end_idx], latent=latent)
					return dKp, dKq
							
			def lcov(self, X, Z, diag=False):
				raise NotImplementedError('latent cov has no hyperparameter theta')

		u, i, k = self._lookup_kernel_and_theta(i)
		dKernel = u.derivateTheta(i)
		return _DerivativeFun(dKernel, self._theta_idx[k], self._theta_idx[k+1])
		
	def derivateX(self):
		'''
		'''
		
		class _DerivativeFunX(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel):
				self.kernels = kernel._kernels
				self.q = kernel._q
				self.theta_idx = kernel._theta_idx
				
			def cov(self, X, theta, diag=False):
				kernels = self.kernels
				q = self.q
				theta_idx = self.theta_idx
				
				n,d = X.shape
				dK = np.zeros((n,n,d)) if diag == False else np.zeros(n*d)
				
				for i in xrange(q):
					start = theta_idx[i]
					end = theta_idx[i+1]
					theta_i = theta[start:end]
					dkernelX = kernels[i].derivateX()
					dK += dkernelX[i].cov(X, theta_i, diag)
				return dK

			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				kernels = self.kernels
				q = self.q
				theta_idx = self.theta_idx
				
				n,d = X.shape
				m = Z.shape[0]
				dK = np.zeros((m,n,d))
				for i in xrange(q):
					start = theta_idx[i]
					end = theta_idx[i+1]
					thetaPi = thetaP[start:end]
					dkernelX = kernels[i].derivateX()
					
					if latent:
						dK += dkernelX.ccov(X, Z, thetaPi, latent=latent)
					else:
						thetaQi = thetaQ[start:end]
						dK += dkernelX.ccov(X, Z, thetaPi, thetaQi, latent=latent)
				return dK
			
			def lcov(self, X, Z=None, diag=False):
				kernels = self.kernels
				q = self.q
				
				xeqz = Z == None
				n,d = X.shape
				if xeqz:
					dK = np.zeros((n,n,d)) if diag == False else np.zeros(n*d)
				else:
					m = Z.shape[0]
					dK = np.zeros((m,n,d))
				for i in xrange(q):
					dKernelX = kernels[i].derivateX()
					dK += dKernelX.lcov(X, Z, diag)
				return dK
							
		fun = _DerivativeFunX(self)
		return fun

	
	def _lookup_kernel_and_param(self, i): 
		''' 
		'''
		q = self._q
		kernels = self._kernels
		offset = 0
		for j in xrange(q):
			if i < offset+kernels[j].nparams:
				return kernels[j], i-offset, j
			offset += kernels[j].nparams
		raise IndexError('Unknown hyperparameter')

	def _lookup_kernel_and_theta(self, i): 
		''' 
		'''
		q = self._q
		kernels = self._kernels
		offset = 0
		for j in xrange(q):
			if i < offset+kernels[j].ntheta:
				return kernels[j], i-offset, j
			offset += kernels[j].ntheta
		raise IndexError('Unknown hyperparameter={0}'.format(i))
	
	def copy(self):
		q = self._q
		kernels = self._kernels
		cp_kernels = np.empty(q, dtype='object')
		for i in xrange(q):
			cp_kernels[i] = kernels[i].copy()
		
		return CompoundKernel(cp_kernels)
			
	class MixedArray(object):
		def __init__(self, array):
			self.array = array
		
		def __len__(self):
			m = len(self.array)
			n = 0
			for i in xrange(m):
				n += len(self.array[i])
			return n
		
		def __getitem__(self, i):
			array, idx = self.__array_idx_at(i)
			return array[idx]
		
		def __setitem__(self, i, value):
			array, idx = self.__array_idx_at(i)
			array[idx] = value

		def __array_idx_at(self, i):
			m = len(self.array)
			offset = 0
			for j in xrange(m):
				if i < offset+len(self.array[j]):
					return self.array[j], i-offset
				offset += len(self.array[j])
			raise IndexError('Unknown hyperparameter')

			return (self.a, i) if i < len(self.a) else (self.b, i-len(self.a))
		
		def __str__(self):
			a = np.empty(0)
			m = len(self.array)
			for i in xrange(m):
				a = np.r_[a, self.array[i]]
			return str(a) 


class MaskedFeatureConvolvedKernel(ConvolvedKernel):
	'''
	todo : param problems
	'''
	
	__slots__ = ('_kernel',
				 '_mask' 	#mask of the used features
				 )
	def __init__(self, kernel, mask):
		
		params = kernel.params
		self._kernel = kernel
		self._mask = mask
		self._params = params
		self._n = len(params)
		self._ntheta = kernel.ntheta
	
	def __str__(self):
		return "MaskedFeatureConvolvedKernel"

	def cov(self, X, theta, diag=False):
		'''
		'''
		mask = self._mask
		kernel = self._kernel
		return kernel.cov(X[:,mask], theta, diag)
	
	def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		Computes the cross covariance matrix cov[f_p(x), f_q(x')]
		Computes the cross covariance matrix cov[f_p(x), u(x')]
		'''
		mask = self._mask
		kernel = self._kernel
		return kernel.ccov(X[:,mask], Z[:,mask], thetaP, thetaQ, latent)
		
	def lcov(self, X, Z=None, diag=False):
		'''
		Computes the covariance matrix cov[u(x), u(x')] uof the latent process u
		'''
		mask = self._mask
		kernel = self._kernel
		if Z == None:
			return kernel.lcov(X[:,mask], diag=diag)
		else:
			return kernel.lcov(X[:,mask], Z[:,mask], diag=diag)
		
	def gradient_cov(self, covGrad, X, theta, diag=False):
		'''
		'''
		mask = self._mask
		kernel = self._kernel
		return kernel.gradient_cov(covGrad, X[:,mask], theta, diag)
		
	def gradient_ccov(self, covGrad, X, Z, thetaP, thetaQ=None, latent=False):
		'''
		'''
		mask = self._mask
		kernel = self._kernel
		return kernel.gradient_ccov(covGrad, X[:,mask], Z[:,mask], thetaP, thetaQ, latent)
		
	def gradient_lcov(self, covGrad, X, Z=None, diag=False):
		mask = self._mask
		kernel = self._kernel
		if Z == None:
			return kernel.gradient_lcov(covGrad, X[:,mask], diag=diag)
		else:
			return kernel.gradient_lcov(covGrad, X[:,mask], Z[:,mask], diag=diag)

	def derivate(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, mask, i):
				self.kernel = kernel
				self.dKernel = kernel.derivate(i)
				self.mask = mask
			
			def cov(self, X, theta, diag=False):
				mask = self.mask
				dKernel = self.dKernel
				return dKernel.cov(X[:,mask], theta, diag)
			
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				mask = self.mask
				dKernel = self.dKernel
				return dKernel.ccov(X[:,mask], Z[:,mask], thetaP, thetaQ, latent)
			
			def lcov(self, X, Z=None, diag=False):
				mask = self.mask
				dkernel = self.dkernel	
				if Z == None:
					return dkernel.lcov(X[:,mask], diag=diag)
				else:
					return dkernel.lcov(X[:,mask], Z[:,mask], diag=diag)

								
		fun = _DerivativeFun(self._kernel, self._mask, i)
		return fun

	def derivateTheta(self, i):
		
		class _DerivativeFun(ConvolvedKernel._IDerivativeFun):
			def __init__(self, kernel, mask, i):
				self.kernel = kernel
				self.dKernel = kernel.derivateTheta(i)
				self.mask = mask
			
			def cov(self, X, theta, diag=False):
				mask = self.mask
				dKernel = self.dKernel
				return dKernel.cov(X[:,mask], theta, diag)
			
			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):
				mask = self.mask
				dKernel = self.dKernel
				return dKernel.ccov(X[:,mask], Z[:,mask], thetaP, thetaQ, latent)
			
			
			def lcov(self, X, Z=None, diag=False):
				raise NotImplementedError('latent cov has no hyperparameter theta')

		if i != 0:
			raise TypeError('Unknown hyperparameter')
			
								
		fun = _DerivativeFun(self._kernel, self._mask, i)
		return fun

	def derivateX(self):
		'''
		'''
		
		class _DerivativeFunX(ConvolvedKernel._IDerivativeFun):	
			def __init__(self, kernel, mask):
				self.dKernelX = kernel.derivateX()
				self.mask = mask
				
			def cov(self, X, theta, diag=False):
				n,d = X.shape
				if diag:
					dKx = np.zeros((n*d))
				else:
					dKx = np.zeros((n,n,d))
				
				mask = self.mask
				dKernelX = self.dKernelX
				dKx[:,:, mask] = dKernelX.cov(X[:,mask], theta, diag) 
				return dKx

			def ccov(self, X, Z, thetaP, thetaQ=None, latent=False):				
				n,d = X.shape
				m = Z.shape[0]
				dKx = np.zeros((m,n,d))
				mask = self.mask
				dKernelX = self.dKernelX
				dKx[:,:, mask] = dKernelX.ccov(X[:,mask], Z[:,mask], thetaP, thetaQ, latent)
				return dKx
			
			def lcov(self, X, Z=None, diag=False):
				n,d = X.shape
				
				mask = self.mask
				dKernelX = self.dKernelX
				if Z == None:
					m = Z.shape[0]
					dKx = np.zeros((m,n,d))
					dKx = dKernelX.lcov(X[:,mask], diag=diag)
				else:
					dKx = np.zeros((n,n,d))
					dKx[:,:, mask] = dKernelX.lcov(X[:,mask], Z[:,mask], diag=diag)
				return dKx
							
		fun = _DerivativeFunX(self._kernel, self._mask)
		return fun
	
	def copy(self):
		cp_kernel = MaskedFeatureConvolvedKernel(self._kernel.copy(), self._mask)
		return cp_kernel


def check_kernel_gradient(kernel, X, Z=None):
	n = kernel.n_params
	
	err = np.zeros(n, dtype=np.float)
	gradK_tilde = approx_Kprime(kernel, X, Z)
	for i in xrange(n):
		deriv = kernel.derivate(i)
		err[i] = np.sum((gradK_tilde[i]-deriv(X,Z))**2)
		
	return err
	
def approx_Kprime(kernel, X, Z=None, epsilon=np.sqrt(np.finfo(float).eps)):
	params = np.copy(kernel.params)
	n = kernel.n_params
	grad = np.empty(n, dtype=np.object)
	
	K0 = kernel(X,Z)
	#print K0
	#kernel.params = np.zeros(n)
	for i in xrange(n):
		kernel.params[i] = params[i]+epsilon
		grad[i] = (kernel(X,Z)-K0) / epsilon
		kernel.params[i] = params[i]
	
	kernel.params = params
	return grad		


def _check_kernel_gradX(kernel, X, Y):
	[n,m] = X.shape
	YY = np.dot(Y,Y.T)
	H = np.random.randn(200, X.shape[1])
	y = np.random.randn(200)
	
	def _likel_fun(x):
		t = time.time()
		Z = x.reshape(n,m)
		K = kernel(Z)
		Kmn = kernel(H,Z)
		
		
		#likel = (np.trace(np.dot(K, YY))) 
		likel = (np.trace(np.dot(np.linalg.inv(K), YY))) + np.log(np.linalg.det(K))
		likel += np.dot(np.dot(y, np.dot(np.dot(Kmn, np.linalg.inv(K)), Kmn.T)),y) 
		#print 'likel_gradX={0}'.format(time.time()-t)
		return likel
	
	def _grad_fun(x):
		t = time.time()
		Z = x.reshape(n,m)
		dX_kernel = kernel.derivateXOld()
		
		t = time.time()
		gradX = np.empty((n,m))
		for i in xrange(n):
			
			gradXn = dX_kernel(Z[i], Z)
			#print 'gu'
			#print dX_kernel(Z[i], Z)
			#t = time.time()
			for j in xrange(m):
				#G = np.zeros((n,n))
				#G[:,i] = G[i,:] = gradXn[:,j]
				#gradX[i,j] = np.trace(np.dot(YY,G))*0.5
				print 'babanna1'
				print  YY[i,i]*gradXn[i,j]
				gradX[i,j] = np.dot(YY[:,i], gradXn[:,j]) - YY[i,i]*gradXn[i,j]
				gradX[i,j] = gradX[i,j]
				
		print 'grad_gradX2={0}'.format(time.time()-t)
		return gradX

	def _grad_fun1(x):
		t = time.time()
		Z = x.reshape(n,m)
		K = kernel(Z)
		Kmn = kernel(H,Z)
		dX_kernel = kernel.derivateX()
		
		t = time.time()
		dX = dX_kernel(Z,Z)*2.0
		dXm = dX_kernel(Z,H)
		dL = -np.dot(np.dot(np.linalg.inv(K), YY), np.linalg.inv(K)) + np.linalg.inv(K)
		dL -= np.dot(np.dot(np.dot(np.dot(np.linalg.inv(K), Kmn.T), np.outer(y,y)), Kmn), np.linalg.inv(K))
		dLm = 2*np.dot(np.dot(np.linalg.inv(K), Kmn.T), np.outer(y,y)).T
		gradX = np.empty((n,m))
		
		for i in xrange(n):
			
			#gradXn = dX_kernel(Z[i], Z)
			#print 'grad_gradX1={0}'.format(time.time()-t)
			#print 'gu'
			#print dX_kernel(Z[i], Z)
			#t = time.time()
			for j in xrange(m):
				#G = np.zeros((n,n))
				#G[:,i] = G[i,:] = gradXn[:,j]
				#gradX[i,j] = np.trace(np.dot(YY,G))
				#gradX[i,j] = np.dot(dX[:,i,j], YY[:,i]) #- YY[i,i]*dX[i,i,j]/2
				#print 'babanna'
				#print   YY[i,i]*dX[i,i,j]/2
				gradX[i,j] = np.dot(dX[:,i,j], dL[:,i])
				gradX[i,j] += np.dot(dXm[:,i,j], dLm[:,i])
		
		return gradX

	def _grad_fun2(x):
		t = time.time()
		Z = x.reshape(n,m)
		K = kernel(Z)
		Kmn = kernel(H,Z)
		
		t = time.time()
		dL = -np.dot(np.dot(np.linalg.inv(K), YY), np.linalg.inv(K)) + np.linalg.inv(K)
		dL -= np.dot(np.dot(np.dot(np.dot(np.linalg.inv(K), Kmn.T), np.outer(y,y)), Kmn), np.linalg.inv(K))
		dLm = 2*np.dot(np.dot(np.linalg.inv(K), Kmn.T), np.outer(y,y)).T
		
		
		gradX = kernel.gradientX(dL, Z)*2.0 + kernel.gradientX(dLm, Z, H)
		return gradX.reshape((n,m))

	_likel_fun(X.ravel())
	#_grad_fun(X.ravel())

	print _likel_fun(X.ravel())
	print spopt.approx_fprime(X.ravel(), _likel_fun, np.sqrt(np.finfo(float).eps)).reshape(n,m)
	#print _grad_fun(X.ravel())
	print _grad_fun1(X.ravel())
	#print 'X'
	print _grad_fun2(X.ravel())
	#print nig_grad_a(np.log(2.0))

def _check_kernel_grad(kernel, X, Y):
	YY = np.dot(Y,Y.T)
	params = np.copy(kernel.params)
	
	def _likel_fun(p):
		
		t = time.time()
		kernel.params = p
		K = kernel(X)
		#iK = np.linalg.inv(K)
		likel = np.trace(np.dot(K, YY))
		#likel = -0.5*np.log(np.linalg.det(K)) #- 0.5*np.dot(np.dot(Y[:,0], iK), Y[:,0])
		#likel = -0.5 * np.dot(np.dot(Y[:,0], iK), Y[:,0])
		
		#print 'likel_grad={0}'.format(time.time()-t)
		
		return likel
	
	def _grad_fun(p):
		kernel.params = p
		d = len(p)
		
		#K = kernel(X)
		#iK = np.linalg.inv(K)
		t = time.time()
		grad = np.zeros(d)
		for j in xrange(d):
			d_kernel = kernel.derivate(j)
			#dK = d_kernel(X,X)
			dK = d_kernel(X)

			grad[j] = np.trace(np.dot(YY,dK))
			#grad[i] = np.trace(np.dot(iK,dK))
			#grad[j] = -0.5*np.trace(np.dot(iK,dK))# - 0.5*np.dot(np.dot(np.dot(Y[:,0], iK), dK), np.dot(iK, Y[:,0]))
			#grad[j] = 0.5*np.dot(np.dot(np.dot(Y[:,0], iK), dK), np.dot(iK, Y[:,0]))
			#grad[j] = -0.5*np.dot(np.dot(Y[:,0], dK), Y[:,0])
		
		#print 'grad_grad={0}'.format(time.time()-t)
		return grad
	
	def _grad_fun1(p):
		kernel.params = p
		
		return kernel.gradient(YY, X)

	_likel_fun(params)
	_grad_fun(params)
	print _likel_fun(params)
	print spopt.approx_fprime(params, _likel_fun, np.sqrt(np.finfo(float).eps))
	print _grad_fun(params)
	print _grad_fun1(params)
	#print nig_grad_a(np.log(2.0))

def _check_convkernel_grad(kernel, X, Y, theta_p=None, theta_q=None):
	YY = np.dot(Y,Y.T)
	params = np.copy(kernel.params)
	
	def _likel_fun(p):
		
		t = time.time()
		kernel.params = p
		#K = kernel.cov(X, theta_p)
		#K = kernel.ccov(X, X, theta_p, latent=True)
		K = kernel.ccov(X, X, theta_p, theta_q)
		#K = kernel.lcov(X)
		#K = K
		#iK = np.linalg.inv(K)
		likel = np.trace(np.dot(K, YY))
		
		#print 'likel_grad={0}'.format(time.time()-t)
		#print 'likel={0}'.format(likel)
		return likel
	
	def _grad_fun(p):
		kernel.params = p
		d = len(p)
		
		#K = kernel(X)
		#iK = np.linalg.inv(K)
		t = time.time()
		grad = np.zeros(d)
		for j in xrange(d):
			d_kernel = kernel.derivate(j)
			#dK = d_kernel.cov(X, theta_p)
			#dK = d_kernel.ccov(X, X, theta_p, latent=True)
			dK = d_kernel.ccov(X, X, theta_p, theta_q)
			#dK = d_kernel.lcov(X)
			grad[j] = np.trace(np.dot(YY,dK))
		
		print 'kernel_grad={0}'.format(grad)
		#print 'kernel_gradA={0}'.format(kernel.gradient_lcov(YY,X))
		#print 'kernel_gradA={0}'.format(kernel.gradient_cov(YY,X,theta_p))
		#print 'kernel_gradA={0}'.format(kernel.gradient_ccov(YY,X,X,theta_p,latent=True))
		print 'kernel_gradA={0}'.format(kernel.gradient_ccov(YY,X,X,theta_p,theta_q))
		#print 'grad_grad={0}'.format(time.time()-t)
		return grad

	_likel_fun(params)
	_grad_fun(params)
	
	print _likel_fun(params)
	print 'approx_grad={0}'.format(spopt.approx_fprime(params, _likel_fun, np.sqrt(np.finfo(float).eps)))
	print 'real_grad={0}'.format(_grad_fun(params))

def _check_convkernel_theta_grad(kernel, X, Y, theta_p=None, theta_q=None):
	YY = np.dot(Y,Y.T)
	params = np.copy(theta_p)
	
	def _likel_fun(p):
		
		t = time.time()
		#kernel.params = p
		#K = kernel.cov(X, p)
		#K = kernel.ccov(X, X, p, latent=True)
		K = kernel.ccov(X, X, theta_q, p)
		#iK = np.linalg.inv(K)
		likel = np.trace(np.dot(K, YY))
		
		#print 'likel_grad={0}'.format(time.time()-t)
		
		return likel
	
	def _grad_fun(p):
		#kernel.params = p
		d = len(p)
		
		print 'plen={0}'.format(len(p))
		
		#K = kernel(X)
		#iK = np.linalg.inv(K)
		t = time.time()
		grad = np.zeros(d)
		for j in xrange(d):
			d_kernel = kernel.derivateTheta(j)
			#dK = d_kernel.cov(X, p)
			K = d_kernel.ccov(X, X, p, latent=True)
			_, dK = d_kernel.ccov(X, X, theta_q, p)

			grad[j] = np.trace(np.dot(YY,dK))
			
		print 'kernel_grad={0}'.format(grad)
		print 'kernel_gradA={0}'.format(kernel.gradient_ccov(YY,X, X, theta_q, p))
		#print 'kernel_gradA={0}'.format(kernel.gradient_ccov(YY,X, X, p, latent=True))
		#print 'kernel_gradA={0}'.format(kernel.gradient_cov(YY,X, p))
		#print 'grad_grad={0}'.format(time.time()-t)
		return grad

	_likel_fun(params)
	_grad_fun(params)
	print _likel_fun(params)
	print 'approx_grad={0}'.format(spopt.approx_fprime(params, _likel_fun, np.sqrt(np.finfo(float).eps)))
	print 'real_grad={0}'.format(_grad_fun(params))


if __name__ == '__main__':
	import numpy as np	
	import scipy as sp
	import time
	
	from upgeo.base.gp import GPRegression
	
	
	P = np.asarray([1,1])
	
	#kernel = SEKernel(np.log(1), np.log(0.1)) + NoiseKernel(np.log(0.5))
	kernel = ARDSEKernel(np.log(12)*np.ones(3), np.log(1)) + NoiseKernel(np.log(0.5))
	print np.copy(kernel.params)
	kernel.params = np.zeros(5)
	print kernel.params
	#kernel = LinearKernel()
	#kernel = BiasedLinearKernel(np.log(2))
	#kernel = ARDLinearKernel(np.log(1)*np.ones(3), np.log(12)) #+ NoiseKernel(np.log(0.5))
	#kernel = PolynomialKernel(1, np.log(14.2), np.log(20))
	#kernel = SEKernel(np.log(1), np.log(0.1)) + LinearKernel() + NoiseKernel(np.log(2))
	#kernel = SEKernel(np.log(1), np.log(0.1)) * SEKernel(np.log(2), np.log(4)) 
	#kernel = SEKernel(np.log(1), np.log(0.1)) + SqConstantKernel(np.log(2))*LinearKernel() + NoiseKernel(np.log(2))
	#kernel = GaussianKernel(np.log(1)*np.ones(3), np.log(5))
	#kernel = GaussianKernel(np.log(1)*np.ones(3))
	kernel_deriv = kernel.derivate(1)
	kernel_derivX = kernel.derivateX()
	
	X =  np.array( [[-0.5046,	0.3999,   -0.5607],
					[-1.2706,   -0.9300,	2.1778],
					[-0.3826,   -0.1768,	1.1385],
					[0.6487,   -2.1321,   -2.4969],
					[0.8257,	1.1454,	0.4413],
					[-1.0149,   -0.6291,   -1.3981],
					[-0.4711,   -1.2038,   -0.2551],
					[0.1370,   -0.2539,	0.1644],
					[-0.2919,   -1.4286,	0.7477],
					[0.3018,   -0.0209,   -0.2730]])
	x1 = np.array([-0.5046,	0.3999, -0.5607])
	
#	X = np.random.randn(3,2)
	Z = np.random.randn(4,3)
#	


	D = np.random.randn(1000, 100)
	kernel = ARDSEKernel(np.log(0.2)*np.ones(100), np.log(2))
	t = time.time()
	kernel(D,D)
	print 'ardkernel time={0}'.format(time.time()-t)
	kernel = ARDSEKernel(np.log(0.7)*np.ones(3), np.log(2))
	print kernel(X)

	print 'gaga'
	kernel = RBFKernel(np.log(1.45), np.log(0.8))
	print kernel(X)
	kernel = ARDRBFKernel(np.log(1.45)*np.ones(3), np.log(0.8))
	print kernel(X)
	kernel = SEKernel(np.log(1.45), np.log(0.8))
	print kernel(X)
	kernel = ARDSEKernel(np.log(1.45)*np.ones(3), np.log(0.8))
	print kernel(X)


	print 'baangabung'
	kernel = LinearKernel()
	dX_kernel = kernel.derivateXOld()
	n,d = X.shape
	print 'kernel X'
	for i in xrange(n):
		print dX_kernel(X[i], Z)
	
	dX_kernel = kernel.derivateX()
	print 'kernel Xi'
	dX = dX_kernel(X,Z)
	for i in xrange(n):
		print dX[:,i,:]
	#print dX 
	#print dX[:,:,1]
	#print dX[1]
	
	
	
	print 'baangabung1'
	kernel = SEKernel(np.log(1), np.log(0.1))
	dX_kernel = kernel.derivateXOld()
	n,d = X.shape
	print 'kernel X'
	for i in xrange(n):
		print dX_kernel(X[i], X)
	
	dX_kernel = kernel.derivateX()
	print 'kernel Xi'
	dX = dX_kernel(X)
	for i in xrange(n):
		print dX[:,i,:]
	#print dX 
	#print dX[:,:,1]
	#print dX[1]	
	
	
	kernel = SEKernel(np.log(1), np.log(0.1)) #NoiseKernel(np.log(0.5)) + LinearKernel()
	kernel = FixedParameterKernel(SEKernel(np.log(1), np.log(0.1))+ NoiseKernel(np.log(0.5)), [1]) 
	#kernel = PolynomialKernel(3, np.log(2), np.log(3))
	#kernel = RBFKernel(np.log(2.45), np.log(0.8))
	#kernel = ARDRBFKernel(np.log(2.45)*np.ones(3), np.log(0.8))
	#kernel = ARDSEKernel(np.log(2.45)*np.ones(3), np.log(0.8))
	#kernel = SqConstantKernel(np.log(5))*LinearKernel()
	#kernel = ARDLinearKernel(np.log(0.32)*np.ones(3), np.log(0.2))
	#kernel = BiasedLinearKernel(np.log(0.0002))
	#kernel = GaussianKernel(np.log(1.2)*np.ones(3))
	#kernel = PolynomialKernel(3, np.log(0.8), np.log(0.4))
	#kernel = ARDPolynomialKernel(1, np.log(1.8)*np.ones(3), np.log(0.8), np.log(0.4))
	#kernel = ARDRBFLinKernel(np.log(1.8)*np.ones(3), np.log(0.8), np.log(0.4))
	Y = np.random.randn(10,8)
	
	print 'gagugrad'
	_check_kernel_grad(kernel, X, Y)
	print 'gradX'
	_check_kernel_gradX(kernel, X, Y)
	
	print 'llalala'
	print kernel(X)
	
	#check group kernel gradients
	Xi = np.c_[np.r_[np.ones(3)*1, np.ones(3)*2, np.ones(4)*10], X]
	#kernel = MaskedFeatureKernel(SEKernel(np.log(1), np.log(0.1)), np.array(np.r_[0, np.ones(3)],dtype=bool)) + CorrelatedNoiseKernel(0, np.log(9), np.log(0.5))#+ NoiseKernel(np.log(0.5))
	kernel = FixedParameterKernel(MaskedFeatureKernel(SEKernel(np.log(1), np.log(0.1)), np.array(np.r_[0, 0,0,1],dtype=bool))*MaskedFeatureKernel(SEKernel(np.log(1), np.log(0.5)), np.array(np.r_[0, 1,1,0],dtype=bool))+ NoiseKernel(np.log(0.5)) + TaskNoiseKernel(Xi[:,0], 0, np.log(0.1)), [1,3])
	print 'mask kernel grad and groupnoise gradient'
	_check_kernel_grad(kernel, Xi, Y)
	print 'gradX'
	_check_kernel_gradX(kernel, Xi, Y)
	
	
	print 'convolved kernel'
	#theta_p = np.r_[np.log(1.4)*np.ones(3), np.log(1)]
	#theta_q = np.r_[np.log(2)*np.ones(3), np.log(0.5)]
	#kernel = ExpGaussianKernel(np.log(1)*np.ones(3))
	
	#kernel = ExpARDSEKernel(np.log(0.8)*np.ones(3), np.log(0.5))
	#kernel.params = np.copy(theta_q)
	#kernel.cov(X, theta_p)
	
	#kernel = ExpGaussianKernel(np.log(0.8))
	#kernel = ExpSEKernel(np.log(0.8), np.log(0.5))
	#kernel = ExpARDGaussianKernel(np.log(1)*np.ones(3))
	#kernel = ExpARDSEKernel(np.log(1)*np.ones(3), np.log(0.5))
	kernel = DiracConvolvedKernel(SEKernel(np.log(0.5), np.log(0.8)))
	#theta_p = np.r_[np.log(0.2), np.log(0.3)]
	#theta_q = np.r_[np.log(1.8), np.log(0.9)]
	#theta_p = np.r_[np.log(0.2), np.log(0.4), np.log(0.3), np.log(0.3)]
	#theta_q = np.r_[np.log(1.8), np.log(1.2), np.log(0.2), np.log(0.9)]
	theta_p = np.r_[np.log(0.2)]
	theta_q = np.r_[np.log(1.8)]
	print 'gadget'
	print 'convgrad'
	_check_convkernel_grad(kernel, X, Y, theta_p, theta_q)
	print 'convthetagrad'
	_check_convkernel_theta_grad(kernel, X, Y, theta_p, theta_q)
	
	print 'buff'
	print kernel.lcov(X)+kernel.lcov(X)
	
	#kernel = ExpARDSEKernel(np.log(0.8)*np.ones(3), np.log(0.5))
	#kernel = CompoundKernel([ExpARDSEKernel(np.log(1)*np.ones(3), np.log(0.5)),ExpARDSEKernel(np.log(0.5)*np.ones(3), np.log(1.5))])
	kernel = CompoundKernel([ExpGaussianKernel(np.log(1)),ExpGaussianKernel(np.log(2))])
	print 'baaff'
	print kernel.lcov(X)
	#theta_p = np.r_[np.log(1.4)*np.ones(3), np.log(1), np.log(0.8), np.log(12), np.log(2.3), np.log(1.5)]
	theta_p = np.copy(np.r_[theta_p, theta_p])
	#theta_q = np.r_[np.log(0.2)*np.ones(3), np.log(0.5), np.log(0.2), np.log(3), np.log(0.3), np.log(2.5)]
	theta_q = np.r_[theta_q, theta_q]
	print 'convgrad'
	_check_convkernel_grad(kernel, X, Y, theta_p, theta_q)
	print 'convthetagrad'
	_check_convkernel_theta_grad(kernel, X, Y, theta_p, theta_q)
	
	
	kernel = ARDSEKernel(np.log(1)*np.ones(3), np.log(5)) #+ NoiseKernel(np.log(0.5))
	print 'psst'
	Z = np.random.randn(200,3)
	Y = np.random.randn(200,8)
	_check_kernel_grad(kernel, Z, Y)
	_check_kernel_gradX(kernel, Z, Y)
	
	print 't'
	print kernel(X,X)
	print 't1'
	print kernel_deriv(X)
	print 't2'
	print kernel_derivX(x1, X)
	
	kernel = ARDSEKernel(np.log(1)*np.ones(3), np.log(5))
	kernel_deriv = kernel.derivate(1)
	kernel_derivX = kernel.derivateX()
	kernel_derivXf = kernel.derivateXf()
	
	Z = np.random.randn(2000,3)
	z1 = np.random.randn(3)
	z2 = np.random.randn(50,3)
	
	t = time.time()
	kernel(Z)
	print 't1={0}'.format(time.time()-t)
	
	t = time.time()
	kernel_deriv(Z)
	print 't2={0}'.format(time.time()-t)
	
	t = time.time()
	kernel_derivX(z1, Z)
	print 't3a={0}'.format(time.time()-t)
	
	t = time.time()
	for i in xrange(50):
		kernel_derivX(z1, Z)
	print 't3b={0}'.format(time.time()-t)
	
	t = time.time()
	kernel_derivXf(z2, Z)
	print 't3b={0}'.format(time.time()-t)
	
	t = time.time()
	R = distance_matrix(Z, Z, metric='sqeuclidean')
	print 't5={0}'.format(time.time()-t)

	t = time.time()
	0.2 * np.exp(-R/(2.0*1.5))
	print 't6={0}'.format(time.time()-t)
	
	t = time.time()
	0.2 * -R/(2.0*1.5)
	print 't7={0}'.format(time.time()-t)
	
	t = time.time()
	0.2 * np.exp(-R/(2.0*1.5), R)
	print 't8={0}'.format(time.time()-t)

	
	
	kernel = ARDSEKernel(np.log(0.5)*np.ones(3), np.log(1))
	print 'kernels'
	print kernel(X)
	kernel = GaussianKernel(np.log(1)*np.ones(3))
	print kernel(X)
	

#	K = fitc_kernel(X)
#	print np.linalg.inv(K+np.eye(10))
#	L = np.linalg.cholesky(K+np.eye(10))
#	print np.dot(np.linalg.inv(L).T, np.linalg.inv(L))
#	t = time.time()
#	#cho_solve((L, 1), np.eye(2000))
#	print 't1={0}'.format(time.time()-t)
#	K = kernel(X)
#	t = time.time()
#	np.linalg.inv(K)
#	print 't2={0}'.format(time.time()-t)
#	
#	
#	
#	print kernel
#	l = 4
#	X = np.asarray([[2,3],[1,2],[4,5],[0.2,1],[0.8,3]])
#	y = np.asarray([2,3,4,5,0.2])
#	print kernel(X) 
#	print kernel_deriv(X)
#	print check_kernel_gradient(kernel, X)
#	
#	gp = GPRegression(kernel)
#	gp.fit(X, y)
#	print gp.log_likel
#	print gp.likel_fun.gradient()
#	
#	#print X.shape
#	#print np.dot(np.diag(1/P),X.T)
#	#K = distance_matrix(np.dot(np.diag(1/P),X.T).T,metric='sqeuclidean')
#	#print np.exp(-K/2)
#	#print kernel(X)
#	
#	#X = X[:,np.newaxis]
#	#gp = GPRegression(kernel)
#	#gp.fit(X, y)
#	#print gp.likel_fun.gradient()
#	#print gp.log_likel
#	
#	i = 0
#	kernel_deriv = kernel.derivate(i)
#	#print kernel_deriv(X)
#	
#	K = distance_matrix(np.dot(np.diag(1/P),X.T).T,metric='cityblock')
#	
#	#print check_kernel_gradient(kernel, X)
#	
#	#print distance_matrix(X[:,i,np.newaxis]/2, metric='sqeuclidean')
#	
	
#	X = np.random.randn(5,2)
#	Y = np.random.randn(10,2)
#	Z = np.r_[X,Y]
#	k = SEKernel(2) * NoiseKernel(4)
#	
#	print k
#	print k.params
#	print k.params[0]
#	#k.params = np.array([3,1,4,4])
#	print k
#	print k.params
#	
#	
#	print 'hallo'
#	print check_kernel_gradient(k, X)
#	
#	
#	
#	
#	
#	kprime = k.derivate(1)
#	
#	
#	approxK = approx_Kprime(k, X)
##	print approxK[1]
##	print kprime(X)
##	print approxK[1]-kprime(X)
##	print sum(sum((approxK[1]-kprime(X))**2))
#	kprime = k.derivate(1)
	
	
#	fprime = lambda x: kprime(np.atleast_2d(x[0]), np.atleast_2d(x[1]))
	
#	f = lambda hyp: k._set_params(hyp); k(X[0], X[1])
	
#	print f([1,2])
	 

	
	
#	from scipy.spatial.distance import cityblock, euclidean, sqeuclidean, minkowski
#	
#	k = SEKernel(8)+NoiseKernel()
#	
#	x = np.random.randn(6)
#	y = np.random.randn(10000)
#	
#	X = np.random.randn(10000,20)
#	K = k(X)
#	
#	
#	t = time.time()
#	K_inv = np.linalg.inv(K)
#	#print K_inv
#	print time.time()-t
#	
#	t = time.time()
#	L = np.linalg.cholesky(K)
#	print time.time()-t
#	K_inv = cho_solve((L,1), np.eye(10000))
#	#print K_inv
#	print time.time()-t
#	
#	a = np.linalg.solve(L, y)
#	a = np.linalg.solve(L.T, a)
#	print a
#	print cho_solve((L, 1), y)
#	
#	Q = np.dot(a,a.T) - K_inv
#	print np.trace(Q)
#	
#	Q = K_inv - np.dot(a,a.T)
#	print np.sum(np.sum(Q))
#	