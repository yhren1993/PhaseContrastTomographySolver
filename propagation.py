"""
Propagation framework functions for AET in Torch

David Ren      david.ren@berkeley.edu

September 16, 2019
"""
import contexttimer
import torch
import torch.nn as nn
import operators as op
from aperture import generate_angular_spectrum_kernel
complex_mul = op.ComplexMul.apply


import numpy as np


class SingleSlicePropagation(torch.autograd.Function):
    '''
    Angular spectrum convolution class for autograd
    inputs:
    field_in: input field at the layer
	pixel_size: sampling voxel size in 3D, should be a 3D volume in (ps_y, ps_x, ps_z)
	wavelength: wavelength of the field
	refractive_index: the index of the media in which the wave propagates (air by defaultyou)
    '''
    @staticmethod
    def forward(ctx, field_in, propagation_distance, phase):
        ctx.save_for_backward(phase)
        ctx.propagation_distance = propagation_distance
        if propagation_distance == 0:
            return field_in
        kernel = op.exp(abs(propagation_distance) * phase)
        kernel = kernel if propagation_distance > 0. else op.conj(kernel)
        field_out = op.convolve_kernel(field_in, kernel, n_dim=2, flag_inplace=False)
        return field_out

    @staticmethod
    def backward(ctx, grad_output):
        phase, = ctx.saved_tensors
        propagation_distance = ctx.propagation_distance
        if propagation_distance == 0:
            return grad_output
        kernel = op.exp(abs(propagation_distance) * phase)
        kernel = kernel if propagation_distance < 0. else op.conj(kernel)
        grad_output = op.convolve_kernel(grad_output, kernel, n_dim=2, flag_inplace=False)
        return grad_output, None, None
propagate = SingleSlicePropagation.apply

class Defocus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, field, kernel, defocus_list= [0.0]):
        ctx.save_for_backward(kernel)
        ctx.defocus_list = defocus_list
        field = torch.fft(field, signal_ndim=2)
        field = field.unsqueeze(2).repeat(1, 1, len(defocus_list), 1).permute(2,0,1,3)
        for defocus_idx in range(len(defocus_list)):
            kernel_temp = op.exp(abs(defocus_list[defocus_idx]) * kernel)
            kernel_temp = kernel_temp if defocus_list[defocus_idx] > 0. else op.conj(kernel_temp)
            field[defocus_idx,...] = op.multiply_complex(field[defocus_idx,...], kernel_temp)
        field = torch.ifft(field, signal_ndim=2).permute(1,2,0,3)
        return field

    @staticmethod
    def backward(ctx, grad_output):
        kernel, = ctx.saved_tensors
        defocus_list = ctx.defocus_list
        grad_output = torch.fft(grad_output.permute(2,0,1,3), signal_ndim=2)
        for defocus_idx in range(len(defocus_list)):
            kernel_temp = op.exp(abs(defocus_list[defocus_idx]) * kernel)
            kernel_temp = kernel_temp if defocus_list[defocus_idx] < 0. else op.conj(kernel_temp)
            grad_output[defocus_idx,...] = op.multiply_complex(grad_output[defocus_idx,...], kernel_temp)
        grad_output = torch.ifft(grad_output, signal_ndim=2).permute(1,2,0,3)
        temp_f =  grad_output.sum(2)
        return grad_output.sum(2), None, None

class MultislicePropagation(nn.Module):
    def __init__(self, shape, voxel_size, wavelength, refractive_index=1.0, dtype=torch.float32, device=torch.device('cuda'), **kwargs):
        super(MultislicePropagation, self).__init__()
        self.shape            = shape       
        self.voxel_size       = voxel_size
        self.wavelength       = wavelength
        self.refractive_index = refractive_index
        self.dtype            = dtype
        self.device           = device
        self.kernel           = generate_angular_spectrum_kernel(self.shape[0:2], self.voxel_size[0], \
                                                                 self.wavelength, refractive_index=self.refractive_index, \
                                                                 numerical_aperture=None, flag_band_limited=False, \
                                                                 dtype=self.dtype, device=self.device)    
        self.distance_to_center = (self.shape[2]/2. - 1/2.) * self.voxel_size[2]
    def forward(self, obj, field_in=None):
        field = field_in
        with contexttimer.Timer() as timer:
            if field is None:
                field = obj[:,:,0,:]
            else:
                field = complex_mul(field, obj[:,:,0,:])
            field = propagate(field, self.voxel_size[2], self.kernel)    
            for layer_idx in range(1, self.shape[2]):
                field = complex_mul(field, obj[:,:,layer_idx,:])
              
                if layer_idx < self.shape[2] - 1:
                    #Propagate forward one layer
                    field = propagate(field, self.voxel_size[2], self.kernel)
                else:
                    field = propagate(field, -1. * self.distance_to_center, self.kernel)
        return field