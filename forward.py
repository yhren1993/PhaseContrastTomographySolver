"""
top level module for pytorch

David Ren      david.ren@berkeley.edu

September 16, 2019
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(precision=10)

#data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.optim as optim
import operators as op
import utilities
from aperture import Pupil
from propagation import SingleSlicePropagation, Defocus, MultislicePropagation
from regularizers import Regularizer

import contexttimer
import scipy.io as sio
import numpy as np

bin_obj       = utilities.BinObject.apply
complex_exp   = op.ComplexExp.apply
complex_mul   = op.ComplexMul.apply
complex_abs   = op.ComplexAbs.apply
field_defocus = Defocus.apply

class PhaseContrastTomography(nn.Module):
	def __init__(self, shape, voxel_size, wavelength, sigma=None, binning_factor=1, pad_size=[0,0], **kwargs):
		#create object -- multiple layers of projected electrostatic potentials
		super(PhaseContrastTomography, self).__init__()
		self.binning_factor = binning_factor
		self.shape          = shape
		self.pad_size       = pad_size
		self.voxel_size     = voxel_size
		self.wavelength     = wavelength
		
		#forward propagation
		self.shape_prop          = self.shape.copy()
		self.shape_prop[2]     //= self.binning_factor
		self.voxel_size_prop     = self.voxel_size.copy()
		self.voxel_size_prop[2] *= self.binning_factor
		self._propagation = MultislicePropagation(self.shape_prop, self.voxel_size_prop, self.wavelength, **kwargs)
		
		self.sigma          = sigma
		if self.sigma is None:
			self.sigma = (2 * np.pi / self.wavelength) * self.voxel_size_prop[2]

		#filter with aperture
		self._pupil       = Pupil(self.shape[0:2], self.voxel_size[0], self.wavelength, **kwargs)

	def forward(self, obj, defocus_list):
		#bin object
		obj = bin_obj(obj, self.binning_factor)
		#raise to transmittance
		obj = complex_exp(complex_mul(op._j, self.sigma * obj))
		#forward propagation & defocus
		field = self._propagation(obj)

		#pupil
		field = self._pupil(field)
		#defocus		
		field = field_defocus(field, self._propagation.kernel, defocus_list)
		#crop
		field = F.pad(field, (0,0,0,0, \
							  -1 * self.pad_size[1], -1 * self.pad_size[1], \
							  -1 * self.pad_size[0], -1 * self.pad_size[0]))
		#compute amplitude
		amplitudes = complex_abs(field)

		return amplitudes

class AETDataset(Dataset):
	def __init__(self, amplitude_measurements=None, tilt_angles=None, defocus_stack=None, **kwargs):
		"""
		Args:
		    transform (callable, optional): Optional transform to be applied
		        on a sample.
		"""
		self.amplitude_measurements = amplitude_measurements
		if self.amplitude_measurements is not None:
			self.amplitude_measurements = amplitude_measurements.astype("float32")
		self.tilt_angles = tilt_angles * 1.0
		self.defocus_stack = defocus_stack * 1.0

	def __len__(self):
		return self.tilt_angles.shape[0]

	def __getitem__(self, idx):
        #X x Y x #defocus
		if self.amplitude_measurements is not None:
			return self.amplitude_measurements[...,idx], self.tilt_angles[idx], self.defocus_stack
		else:
			return self.tilt_angles[idx], self.defocus_stack

class TorchTomographySolver:
	def __init__(self, **kwargs):
		"""
		Creating tomography solver object.
		Required Args:
			shape: shape of the object in [y, x, z]
			voxel_size: size of voxel in [y, x, z]
			wavelength: wavelength of probing wave, scalar
			sigma: sigma used in calculating transmittance function (exp(1i * sigma * object)), scalar
			tilt_angles: an array of sample rotation angles
			defocus_stack: an array of defocus values

		Optional Args [default]
			amplitude_measurements: measurements for reconstruction, not needed for forward evaluation of the model only [None]
			numerical_aperture: numerical aperture of the system, scalar [1.0]
			binning_factor: bins the number of slices together to save computation, scalar [1]
			pad_size: padding reconstruction from measurements in [dy,dx], final size will be measurement.shape + 2*[dy, dx], [0, 0]
			batch_size: reconstruction batch size, scalar [1]
			shuffle: random shuffle of measurements, boolean [True]
			pupil: inital value for the pupil function [None]
			refractive_index: background refractive index (air 1.0, water 1.33) [1.0]
			maxitr: maximum number of iterations [100]
			step_size: step_size for each gradient update [0.1]
			momentum: [0.0 NOTIMPLEMENTED]

			-- regularizer parameters --
			regularizer_total_variation: boolean [False]
			regularizer_total_variation_gpu: boolean [False]
			regularizer_total_variation_parameter: controls amount of total variation, scalar [1.0]
			regularizer_total_variation_maxitr: number of iterations for total variation, integer [15]
			regularizer_total_variation_order: differential order, scalar [1], higher order not yet implemented
			regularizer_pure_real: boolean [False]
			regularizer_pure_imag: boolean [False]
			regularizer_pure_amplitude: boolean [False]
			regularizer_pure_phase: boolean [False]
			regularizer_positivity_real: boolean [False]
			regularizer_positivity_imag: boolean [False]
			regularizer_negativity_real: boolean [False]
			regularizer_negativity_imag: boolean [False]
			regularizer_dtype: torch dtype class [torch.float32]
		"""
		
		self.shape 			     = kwargs.get("shape")
		
		self.batch_size          = kwargs.get("batch_size",           1)
		self.shuffle		     = kwargs.get("shuffle",              True)
		self.optim_max_itr       = kwargs.get("maxitr",               100)
		self.optim_step_size     = kwargs.get("step_size",            0.1)
		self.optim_momentum      = kwargs.get("momentum",             0.0)

		self.dataset      	     = AETDataset(**kwargs)
		self.aet_obj             = PhaseContrastTomography(**kwargs)
		self.regularizer_obj     = Regularizer(**kwargs)
		self.rotation_obj	     = utilities.ImageRotation(self.shape, axis = 0)
		
		self.cost_function       = nn.MSELoss(reduction='sum')
    	
	def run(self, obj_init=None, forward_only=False, callback=None):
		"""
		run tomography solver
		Args:
		forward_only: True  -- only runs forward model on estimated object
					  False -- runs reconstruction
		"""
		if forward_only:
			assert obj_init is not None
			self.shuffle = False
			amplitude_list = []
		
		self.dataloader  = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

		error = []
    	#initialize object
		self.obj = obj_init
		if self.obj is None:
			self.obj = op.r2c(torch.zeros(self.shape).cuda())
		else:
			if not self.obj.is_cuda:
				self.obj = self.obj.cuda()
			if len(self.obj.shape) == 3:
				self.obj = op.r2c(self.obj)
		
		#begin iteration
		for itr_idx in range(self.optim_max_itr):
			running_cost = 0.0
			for data_idx, data in enumerate(self.dataloader, 0):
				with contexttimer.Timer() as timer:
		    		
		    		#parse data
					if not forward_only:
						amplitudes, rotation_angle, defocus_list = data
						amplitudes = torch.squeeze(amplitudes)

					else:
						rotation_angle, defocus_list = data
					rotation_angle = rotation_angle.item()
					defocus_list = torch.squeeze(defocus_list)						
					
					#rotate object
					if data_idx == 0:
						self.obj = self.rotation_obj.forward(self.obj, rotation_angle)
					else:
						if abs(rotation_angle - previous_angle) > 90:
							self.obj = self.rotation_obj.forward(self.obj, -1 * previous_angle)
							self.obj = self.rotation_obj.forward(self.obj, rotation_angle)
						else:
							self.obj = self.rotation_obj.forward(self.obj, rotation_angle - previous_angle)					
					
					if not forward_only:
						#define optimizer
						self.obj.requires_grad_()
						optimizer = optim.SGD([self.obj], lr=self.optim_step_size)
					
					#forward scattering
					estimated_amplitudes = self.aet_obj(self.obj, defocus_list)
					if not forward_only:
			    		#compute cost
						cost = self.cost_function(estimated_amplitudes, amplitudes.cuda())
						running_cost += cost.item()

						#backpropagation
						cost.backward()

						#update object
						optimizer.step()
						optimizer.zero_grad()
						del cost
					else:
						#store measurement
						amplitude_list.append(estimated_amplitudes.cpu().detach())
					del estimated_amplitudes
					self.obj.requires_grad = False
					previous_angle = rotation_angle
					
					#rotate object back
					if data_idx == (self.dataset.__len__() - 1):
						previous_angle = 0.0
						self.obj = self.rotation_obj.forward(self.obj, -1.0*rotation_angle)
					print("Rotation {:03d}/{:03d}.".format(data_idx+1, self.dataset.__len__()), end="\r")
			
			#apply regularization
			torch.cuda.empty_cache()
			self.obj = self.regularizer_obj.apply(self.obj)
			error.append(running_cost)
			if callback is not None:
				callback(self.obj.cpu().detach(), error)
			if forward_only and itr_idx == 0:
				return amplitude_list

		return self.obj, error



