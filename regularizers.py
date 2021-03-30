"""
Regularization module for pytorch

David Ren      david.ren@berkeley.edu

September 16, 2019
"""

import numpy as np
import operators as op
import torch
import gc


class Regularizer:
	"""
	Highest-level Regularizer class that is responsible for parsing user arguments to create proximal operators
	All proximal operators operate on complex variables (real & imaginary part separately)
	Pure Amplitude:
        pure_amplitude: boolean, whether or not to enforce object to be purely amplitude
	Pure phase:
        pure_phase: boolean, whether or not to enforce object to be purely phase object        
	Pure Real:
		pure_real: boolean, whether or not to enforce object to be purely real
	
	Pure imaginary:
		pure_imag: boolean, whether or not to enforce object to be purely imaginary

	Positivity:
		positivity_real(positivity_imag): boolean, whether or not to enforce positivity for real(imaginary) part

	Negativity:
		negativity_real(negativity_imag): boolean, whether or not to enforce negativity for real(imaginary) part		
	

	Total variation (2D & 3D):
		total_variation: boolean, whether or not to use total variation regularization
		total_variation_gpu: boolean, whether or not to use GPU implementation
		total_variation_parameter: scalar, regularization parameter (lambda)
		total_variation_maxitr: integer, number of each iteration for total variation		
	"""
	def __init__(self, verbose = True, **kwargs):
		#Given all parameters, construct all proximal operators
		self.prox_list = []
		#Total Variation
		if kwargs.get("regularizer_total_variation", False):
			if kwargs.get("regularizer_total_variation_gpu", False):
				kwargs["regularizer_device"] = torch.device('cuda')
			else:				
				kwargs["regularizer_device"] = torch.device('cpu')
			if kwargs.get("regularizer_total_variation_anisotropic", False):
				self.prox_list.append(TotalVariationAnisotropic(**kwargs))
			else:
				self.prox_list.append(TotalVariation(**kwargs))
		#Others
		else:
			#Purely real
			if kwargs.get("regularizer_pure_real", False):
				self.prox_list.append(PureReal())

			#Purely imaginary
			if kwargs.get("regularizer_pure_imag", False):
				self.prox_list.append(Pureimag())

			#Purely amplitude object
			if kwargs.get("regularizer_pure_amplitude", False):
				self.prox_list.append(PureAmplitude())
	        
			#Purely phase object
			if kwargs.get(("regularizer_pure_phase"), False):
				self.prox_list.append(PurePhase())
            
			#Positivity
			positivity_real = kwargs.get("regularizer_positivity_real", False)
			positivity_imag = kwargs.get("regularizer_positivity_imag", False)
			if positivity_real or positivity_imag:
				self.prox_list.append(Positivity(positivity_real, positivity_imag))

			#Negativity
			negativity_real = kwargs.get("regularizer_negativity_real", False)
			negativity_imag = kwargs.get("regularizer_negativity_imag", False)
			if negativity_real or negativity_imag:
				self.prox_list.append(Negativity(negativity_real, negativity_imag))

		if verbose:
			for prox_op in self.prox_list:
				print("Regularizer -", prox_op.proximal_name)

	def compute_cost(self, x):
		cost = 0.0
		for prox_op in self.prox_list:
			cost_temp = prox_op.compute_cost(x)
			if cost_temp != None:
				cost += cost_temp
		return cost

	def apply(self, x):
		for prox_op in self.prox_list:
			x = prox_op.compute_prox(x)
		torch.cuda.empty_cache()
		return x

class ProximalOperator():
	def __init__(self, proximal_name):
		self.proximal_name = proximal_name
		self.itr_count = 0
	def compute_cost(self):
		pass
	def compute_prox(self):
		pass	
	def set_parameter(self):
		pass
	def _bound_real_value(self, x, value = 0):
		return torch.clamp(x, min=value)

class TotalVariation(ProximalOperator):
	def __init__(self, **kwargs):
		proximal_name       = "Total Variation"

		self.parameter_list = None
		parameter           = kwargs.get("regularizer_total_variation_parameter", 1.0)
		if not np.isscalar(parameter):
			self.parameter_list = parameter
			parameter = self.parameter_list[0]
		self.aniso_ratio    = kwargs.get("regularizer_tv_aniso_ratio",           1)
		maxitr              = kwargs.get("regularizer_total_variation_maxitr",   15)
		self.order          = kwargs.get("regularizer_total_variation_order",    1)
		self.pure_real      = kwargs.get("regularizer_pure_real",                False)
		self.pure_imag      = kwargs.get("regularizer_pure_imag",                False)
		self.pure_amplitude = kwargs.get("regularizer_pure_amplitude",           False)
		self.pure_phase     = kwargs.get("regularizer_pure_phase",               False)        
		self.dtype          = kwargs.get("regularizer_dtype",                    torch.float32)
		self.device         = kwargs.get("regularizer_device",                   torch.device('cuda'))        

		#real part
		if kwargs.get("regularizer_positivity_real", False):
			self.realProjector = lambda x: self._bound_real_value(x, 0)
			proximal_name      = "%s+%s" % (proximal_name, "positivity_real")
		elif kwargs.get("regularizer_negativity_real", False):
			self.realProjector = lambda x: -1.0 * self._bound_real_value(-1.0 * x, 0)
			proximal_name      = "%s+%s" % (proximal_name, "negativity_real")
		else:
			self.realProjector = lambda x: x

		#imaginary part
		if kwargs.get("regularizer_positivity_imag", False):
			self.imagProjector = lambda x: self._bound_real_value(x, 0)
			proximal_name      = "%s+%s" % (proximal_name, "positivity_imag")
		elif kwargs.get("regularizer_negativity_imag", False):
			self.imagProjector = lambda x: -1.0 * self._bound_real_value(-1.0 * x, 0)
			proximal_name      = "%s+%s" % (proximal_name, "negativity_imag")
		else:
			self.imagProjector = lambda x: x
		self.set_parameter(parameter, maxitr)
		super().__init__(proximal_name)
	
	def set_parameter(self, parameter=None, maxitr=None):
		if parameter is not None:
			self.parameter = parameter
		if maxitr is not None:
			self.maxitr = maxitr
		return

	def compute_cost(self, x):
		return None
	
	def compute_prox(self, x):
		if self.parameter_list is not None:
			self.set_parameter(self.parameter_list[self.itr_count])
		x_device = x.device
		x = x.to(device=self.device)
		if self.pure_real:
			x = self._compute_prox_real(x.real, self.realProjector) + 0j
		elif self.pure_imag:
			x = 1j * self._compute_prox_real(x.imag, self.realProjector)
		elif self.pure_amplitude:
			x = op.r2c(self._compute_prox_real(torch.abs(x), self.realProjector))
		elif self.pure_phase:
			x = torch.exp(1j* self._compute_prox_real(torch.angle(x), self.realProjector))
		else:
			x_real = self._compute_prox_real(x.real, self.realProjector)
			self.set_parameter(self.parameter / 1.0, self.maxitr)
			x = x_real + 1j * self._compute_prox_real(x.imag, self.imagProjector)
			self.set_parameter(self.parameter * 1.0, self.maxitr)
		self.itr_count += 1	
		return x.to(x_device)

	def _compute_tv_norm(self, x):
			x_norm             = x**2
			x_norm  		   = x_norm.sum(3 if len(x.shape) == 4 else 2)**0.5
			x_norm	= torch.clamp(x_norm, min=1.0)
			return x_norm		

	def _filter_d(self, x, axis):
		assert axis<3, "This function only supports matrix up to 3 dimension!"
		if self.order == 1:
			if axis == 0:
				Dx     = x - torch.roll(x, 1, dims=0)
			elif axis == 1:
				Dx     = x - torch.roll(x, 1, dims=1)
			else:	
				Dx     = x - torch.roll(x, 1, dims=2)
		else:
			raise NotImplementedError("filter orders larger than 1 are not implemented!")			
		return Dx

	def _filter_dt(self, x):
		if self.order == 1:
			if len(x.shape) == 3:
				DTx    = x[..., 0] - torch.roll(x[..., 0], -1, dims=0) + \
			             x[..., 1] - torch.roll(x[..., 1], -1, dims=1)
			elif len(x.shape) == 4:
				DTx    = x[..., 0] - torch.roll(x[..., 0], -1, dims=0) + \
			             x[..., 1] - torch.roll(x[..., 1], -1, dims=1) + \
			             x[..., 2] - torch.roll(x[..., 2], -1, dims=2)
		else:
			raise NotImplementedError("filter orders larger than 1 are not implemented!")
		return DTx

	def _compute_prox_real(self, x, projector):
		t_k        = 1.0
		
		def _update_gradient():
		    grad_u_hat = x - self.parameter * self._filter_dt(u_k1)
		    return grad_u_hat
		
		u_k  = torch.zeros(x.shape + (3 if len(x.shape) == 3 else 2,), dtype=self.dtype, device=self.device)
		u_k1 = torch.zeros(x.shape + (3 if len(x.shape) == 3 else 2,), dtype=self.dtype, device=self.device)

		for iteration in range(self.maxitr):
			if iteration > 0:
				grad_u_hat  = _update_gradient()
			else:
				grad_u_hat  = x.clone()

			grad_u_hat         = projector(grad_u_hat)
			if len(x.shape) == 2: #2D case
				constant_scale = 8.0
			elif len(x.shape) == 3: #3D case
				constant_scale = 12.0
			u_k1[..., 0] = u_k1[..., 0] + (1.0/constant_scale**self.order/self.parameter) * self._filter_d(grad_u_hat, axis=0)
			u_k1[..., 1] = u_k1[..., 1] + (1.0/constant_scale**self.order/self.parameter) * self._filter_d(grad_u_hat, axis=1)			
			if len(x.shape) == 3: #3D case
				u_k1[..., 2] = u_k1[..., 2] + (1.0/constant_scale**self.order/self.parameter) * self._filter_d(grad_u_hat, axis=2)
			grad_u_hat         = None
			u_k1_norm          = self._compute_tv_norm(u_k1)
			u_k1              /= u_k1_norm.unsqueeze(-1)
			u_k1_norm		   = None
			t_k1               = 0.5 * (1.0 + (1.0 + 4.0*t_k**2)**0.5)
			beta               = (t_k - 1.0)/t_k1
			
			temp = u_k[...,0].clone()
			if iteration < self.maxitr - 1:
				u_k[...,0] = u_k1[...,0]
			u_k1[...,0] =  (1.0 + beta)*u_k1[...,0] - beta*temp #now u_hat
			temp[:] = u_k[...,1]
			if iteration < self.maxitr - 1:
				u_k[...,1] = u_k1[...,1]
			u_k1[...,1] =  (1.0 + beta)*u_k1[...,1] - beta*temp
			if len(x.shape) == 3: #2D case
				temp[:] = u_k[...,2]
				if iteration < self.maxitr - 1:
					u_k[...,2] = u_k1[...,2]
				u_k1[...,2] =  (1.0 + beta)*u_k1[...,2] - beta*temp
			temp = None

		grad_u_hat = projector(_update_gradient())
		u_k 	   = None
		u_k1 	   = None		
		return grad_u_hat

class TotalVariationAnisotropic(TotalVariation):
	"""
	Anisotropic version of TV, meant for 3D only!
	Saves memory comparing to iterative version of TV
	"""
#	def _compute_prox_real(self, x, projector):
#		for iteration in range(self.maxitr):
#			x = self._compute_prox_real_single_iteration(x, projector)
#			torch.cuda.empty_cache()
#		return x
	def _compute_prox_real(self, x, projector):
		assert len(x.shape) == 3
		# parallel proximal method
		for iteration in range(self.maxitr):
			x = projector((1/6)*(self._computeProxRealSingleAxis(x) + \
					   			self._computeProxRealSingleAxis(x,shift=True) + \
					   			self._computeProxRealSingleAxis(x.permute(1,0,2),shift=False).permute(1,0,2) + \
					   			self._computeProxRealSingleAxis(x.permute(1,0,2),shift=True).permute(1,0,2) + \
					   			self._computeProxRealSingleAxis(x.permute(2,0,1),shift=False,parameter=self.parameter * self.aniso_ratio).permute(1,2,0) + \
					   			self._computeProxRealSingleAxis(x.permute(2,0,1),shift=True,parameter=self.parameter * self.aniso_ratio).permute(1,2,0)))
		return x
	def _computeProxRealSingleAxis(self,x_in,shift=False,parameter=None):
		self.Np = x_in.shape
		if np.mod(self.Np[0],2) == 1:
			raise NotImplementedError('Shape cannot be odd')
		if shift:
			x = x_in.roll(1, dims = 0)
		else:
			x = x_in.clone()
		c = torch.from_numpy(np.asarray([1/np.sqrt(2)])).float().to(self.device)
		z1 = self.softThr(c*(x[1::2,:]-x[0::2,:]),parameter)*c
		x[0::2,:] += x[1::2,:]
		x[1::2,:]  = x[0::2,:]
		x         *= c**2
		x[0::2,:] -= z1
		x[1::2,:] += z1
		if shift:
			x = x.roll(-1, dims = 0)
		return x

	def softThr(self,x,parameter):
		if parameter is None:
			parameter = self.parameter
		return torch.sign(x) * (torch.abs(x) - self.parameter) * (torch.abs(x) > self.parameter).float()

class Positivity(ProximalOperator):
	"""Enforce positivity constraint on a complex variable's real & imaginary part."""
	def __init__(self, positivity_real, positivity_imag, proximal_name = "Positivity"):
		super().__init__(proximal_name)
		self.real = positivity_real
		self.imag = positivity_imag

	def compute_cost(self, x):
		return None

	def compute_prox(self, x):
		x_real = torch.real(x)
		x_imag = torch.imag(x)
		if self.real:
			x_real = self._bound_real_value(x_real, 0, self.real)
		if self.imag:
			x_imag = self._bound_real_value(x_imag, 0, self.imag)
		return x_real + 1j * x_imag

class Negativity(Positivity):
	"""Enforce positivity constraint on a complex variable's real & imaginary part."""
	def __init__(self, negativity_real, negativity_imag):
		super().__init__(negativity_real, negativity_imag, "Negativity")

	def compute_prox(self, x):
		return (-1.) * super().compute_prox((-1.) * x)

class PureReal(ProximalOperator):
	"""Enforce real constraint on a complex, imaginary part will be cleared"""
	def __init__(self):
		super().__init__("Pure real")

	def compute_cost(self, x):
		return None

	def compute_prox(self, x):	
		return x.real + 0j

class Pureimag(ProximalOperator):
	"""Enforce imaginary constraint on a complex, real part will be cleared"""	
	def __init__(self):
		super().__init__("Pure imaginary")

	def compute_cost(self, x):
		return None

	def compute_prox(self, x):
		return 1j * x.imag

class PureAmplitude(ProximalOperator):
	def __init__(self):
		super().__init__("Purely Amplitude")    
	def compute_cost(self, x):
		return None
	def compute_prox(self, x):	
		return torch.abs(x)

class PurePhase(ProximalOperator):
	def __init__(self):
		super().__init__("Purely Phase")    
	def compute_cost(self, x):
		return None
	def compute_prox(self, x):	
		return torch.exp(1j * torch.angle(x))

# class Lasso(ProximalOperator):
# 	"""||x||_1 regularizer, soft thresholding with certain parameter"""
# 	def __init__(self, parameter):	
# 		super().__init__("LASSO")
# 		self.set_parameter(parameter)

# 	def _softThreshold(self, x):
# 		if type(x).__module__ == "arrayfire.array":
# 			#POTENTIAL BUG: af.sign implementation does not agree with documentation
# 			x = (af.sign(x)-0.5)*(-2.0) * (af.abs(x) - self.parameter) * (af.abs(x) > self.parameter)
# 		else:
# 			x = np.sign(x) * (np.abs(x) - self.parameter) * (np.abs(x) > self.parameter)
# 		return x

# 	def setParameter(self, parameter):		
# 		self.parameter = parameter

# 	def compute_cost(self, x):
# 		return af.norm(af.moddims(x, np.prod(x.shape)), norm_type = af.NORM.VECTOR_1)

# 	def compute_prox(self, x):	
# 		if type(x).__module__ == "arrayfire.array":
# 			x = self._softThreshold(af.real(x)) + 1.0j * self._softThreshold(af.imag(x))
# 		else:
# 		    x = self._softThreshold(x.real) + 1.0j * self._softThreshold(x.imag)
# 		return x		

