"""
top level module for pytorch

David Ren      david.ren@berkeley.edu

September 16, 2019
"""

import sys
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
import shift
from aperture import Pupil
from propagation import SingleSlicePropagation, Defocus, MultislicePropagation
from regularizers import Regularizer

import scipy.io as sio
import numpy as np
bin_obj       = utilities.BinObject.apply
complex_exp   = op.ComplexExp.apply
complex_mul   = op.ComplexMul.apply
complex_abs   = op.ComplexAbs.apply
#field_defocus = Defocus.apply
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
			defocus_list: an array of defocus values

		Optional Args [default]
			amplitude_measurements: measurements for reconstruction, not needed for forward evaluation of the model only [None]
			numerical_aperture: numerical aperture of the system, scalar [1.0]
			binning_factor: bins the number of slices together to save computation, scalar [1]
			pad_size: padding reconstruction from measurements in [dy,dx], final size will be measurement.shape + 2*[dy, dx], [0, 0]
			shuffle: random shuffle of measurements, boolean [True]
			pupil: inital value for the pupil function [None]
			maxitr: maximum number of iterations [100]
			step_size: step_size for each gradient update [0.1]
			momentum: [0.0 NOTIMPLEMENTED]

			-- Shift alignment parameters -- 
			shift_align: whether to turn on alignment, boolean, [False]
			sa_method: shift alignment method, can be "gradient", "hybrid_correlation", "cross_correlation", or "phase_correlation", string, ["gradient"]
			sa_step_size: step_size of shift parameters, float, [0.1]
			sa_start_iteration: alignment process will not start until then, int, [0]

			-- Defocus refinement parameters -- 
			defocus_refine: whether to turn on defocus refinement for each measurement, boolean, [False]
			dr_method: defocus refinement method, can be "gradient", string, ["gradient"]
			dr_step_size: step_size of defocus refinement parameters, float, [0.1]
			dr_start_iteration: refinement process will not start until then, int, [0]

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
		
		self.shuffle		     = kwargs.get("shuffle",              True)
		self.optim_max_itr       = kwargs.get("maxitr",               100)
		self.optim_step_size     = kwargs.get("step_size",            0.1)
		self.optim_momentum      = kwargs.get("momentum",             0.0)

		#parameters for shift alignment
		self.shift_align         = kwargs.get("shift_align",          False)
		self.sa_method           = kwargs.get("sa_method",            "gradient")
		self.sa_step_size        = kwargs.get("sa_step_size",         0.1)
		self.sa_start_iteration  = kwargs.get("sa_start_iteration",   0)

		#parameters for defocus refinement
		self.defocus_refine      = kwargs.get("defocus_refine",       False)
		self.dr_method           = kwargs.get("dr_method",            "gradient")
		self.dr_step_size        = kwargs.get("dr_step_size",         0.1)
		self.dr_start_iteration  = kwargs.get("dr_start_iteration",   0)		

		if not shift.is_valid_method(self.sa_method):
			raise ValueError('Shift alignment method not valid.')
		if self.shift_align and shift.is_correlation_method(self.sa_method):
			self.shift_obj		 = shift.ImageShiftCorrelationBased(kwargs["amplitude_measurements"].shape[0:2], \
										    					    upsample_factor = 10, method = self.sa_method, \
											 					    device=torch.device('cpu'))

		self.dataset      	     = AETDataset(**kwargs)
		self.num_defocus	     = len(self.dataset.defocus_list)
		self.num_rotation        = len(self.dataset.tilt_angles)
		self.tomography_obj      = PhaseContrastScattering(**kwargs)
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
		
		self.dataloader = DataLoader(self.dataset, batch_size = 1, shuffle=self.shuffle)

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
		
		#initialize shift parameters
		self.yx_shifts = None
		if self.shift_align:
			self.yx_shifts = torch.zeros((2, self.num_defocus, self.num_rotation))

		#begin iteration
		for itr_idx in range(self.optim_max_itr):
			sys.stdout.flush()
			running_cost = 0.0
			for data_idx, data in enumerate(self.dataloader, 0):
	    		#parse data
				if not forward_only:
					amplitudes, rotation_angle, defocus_list, rotation_idx = data
					amplitudes = torch.squeeze(amplitudes)
					if len(amplitudes.shape) < 3:
						amplitudes = amplitudes.unsqueeze(-1)

				else:
					rotation_angle, defocus_list, rotation_idx = data[-3:]
				#prepare tilt specific parameters
				defocus_list = torch.flatten(defocus_list).cuda()
				rotation_angle = rotation_angle.item()
				yx_shift = None
				if self.shift_align and self.sa_method == "gradient" and itr_idx >= self.sa_start_iteration:
					yx_shift = self.yx_shifts[:,:,rotation_idx]
					yx_shift = yx_shift.cuda()
					yx_shift.requires_grad_()
				if self.defocus_refine and self.dr_method == "gradient" and itr_idx >= self.dr_start_iteration:
					defocus_list.requires_grad_()					
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
					optimizer_params = [{'params': self.obj, 'lr': self.optim_step_size}]
					if self.shift_align and self.sa_method == "gradient" and itr_idx >= self.sa_start_iteration:
						optimizer_params.append({'params': yx_shift, 'lr': self.sa_step_size})
					if self.defocus_refine and self.dr_method == "gradient" and itr_idx >= self.dr_start_iteration:
						optimizer_params.append({'params': defocus_list, 'lr': self.dr_step_size})
					optimizer = optim.SGD(optimizer_params)
				
				#forward scattering
				estimated_amplitudes = self.tomography_obj(self.obj, defocus_list, yx_shift)

				#Correlation based shift estimation
				if self.shift_align and shift.is_correlation_method(self.sa_method) and itr_idx >= self.sa_start_iteration:
					amplitudes, yx_shift, _ = self.shift_obj.estimate(estimated_amplitudes, amplitudes)
					yx_shift = yx_shift.unsqueeze(-1)
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
				if self.shift_align and itr_idx >= self.sa_start_iteration:
					yx_shift.requires_grad = False
					self.yx_shifts[:,:,rotation_idx] = yx_shift[:].cpu()
				if self.defocus_refine and itr_idx >= self.dr_start_iteration:
					defocus_list.requires_grad = False
					self.dataset.update_defocus_list(defocus_list[:].cpu(), rotation_idx)

				previous_angle = rotation_angle
				
				#rotate object back
				if data_idx == (self.dataset.__len__() - 1):
					previous_angle = 0.0
					self.obj = self.rotation_obj.forward(self.obj, -1.0*rotation_angle)
				print("Rotation {:03d}/{:03d}.".format(data_idx+1, self.dataset.__len__()), end="\r")
			
			#apply regularization
			amplitudes = None
			torch.cuda.empty_cache()
			self.obj = self.regularizer_obj.apply(self.obj)
			error.append(running_cost)
			if callback is not None:
				callback(self.obj.cpu().detach(), error)
			if forward_only and itr_idx == 0:
				return amplitude_list
			print("Iteration {:03d}/{:03d}. Error: {:03f}".format(itr_idx+1, self.optim_max_itr, np.log10(running_cost)))

		self.defocus_list = self.dataset.get_all_defocus_lists()
		return self.obj.cpu().detach(), error

class AETDataset(Dataset):
	def __init__(self, amplitude_measurements=None, tilt_angles=[0], defocus_list=None, **kwargs):
		"""
		Args:
		    transform (callable, optional): Optional transform to be applied
		        on a sample.
		"""
		self.amplitude_measurements = amplitude_measurements
		if self.amplitude_measurements is not None:
			self.amplitude_measurements = amplitude_measurements.astype("float32")
		if tilt_angles is not None:
			self.tilt_angles = tilt_angles * 1.0
		if defocus_list is not None:
			self.defocus_list = torch.tensor(defocus_list).unsqueeze(1).repeat(1, len(self.tilt_angles)) * 1.0

	def __len__(self):
		return self.tilt_angles.shape[0]

	def __getitem__(self, idx):
        #X x Y x #defocus
		if self.amplitude_measurements is not None:
			return self.amplitude_measurements[...,idx], self.tilt_angles[idx], self.defocus_list[:,idx], idx
		else:
			return self.tilt_angles[idx], self.defocus_list[:,idx], idx

	def update_defocus_list(self,defocus_list, idx):
		self.defocus_list[:,idx] = defocus_list.unsqueeze(-1)
		return

	def get_all_defocus_lists(self):
		return self.defocus_list  


class PhaseContrastScattering(nn.Module):

	def __init__(self, shape, voxel_size, wavelength, sigma=None, binning_factor=1, pad_size=[0,0], **kwargs):
		"""
		Phase contrast scattering model
		Starts from a plane wave, 3D object, and a list of defocus distance (in Angstrom).
		Computes intensity phase contrast image after electron scatters through the sample using multislice algorithm
		Required Args:
			shape: shape of the object in [y, x, z]
			voxel_size: size of voxel in [y, x, z]
			wavelength: wavelength of probing wave, scalar

		Optional Args [default]:
			sigma: sigma used in calculating transmittance function (exp(1i * sigma * object)), scalar [None]
			binning_factor: bins the number of slices together to save computation (loses accuracy), scalar [1]
			pad_size: padding reconstruction from measurements in [dy,dx], final size will be measurement.shape + 2*[dy, dx], [0, 0]
		"""
		super(PhaseContrastScattering, self).__init__()
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
		self._pupil = Pupil(self.shape[0:2], self.voxel_size[0], self.wavelength, **kwargs)

		
		#defocus operator
		self._defocus = Defocus()

		#shift correction
		self._shift = shift.ImageShiftGradientBased(self.shape[0:2], **kwargs)

	def forward(self, obj, defocus_list, yx_shift=None):
		#bin object
		obj = bin_obj(obj, self.binning_factor)
		#raise to transmittance
		obj = complex_exp(complex_mul(op._j, self.sigma * obj))
		#forward propagation & defocus
		field = self._propagation(obj)
		#pupil
		field = self._pupil(field)
		#defocus		
		field = self._defocus(field, self._propagation.propagate.kernel_phase, defocus_list)
		#shift
		field = self._shift(field, yx_shift)
		#crop
		field = F.pad(field, (0,0,0,0, \
							  -1 * self.pad_size[1], -1 * self.pad_size[1], \
							  -1 * self.pad_size[0], -1 * self.pad_size[0]))
		#compute amplitude
		amplitudes = complex_abs(field)

		return amplitudes







