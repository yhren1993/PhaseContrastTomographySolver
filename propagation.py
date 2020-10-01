"""
Propagation framework functions for AET in Torch

David Ren      david.ren@berkeley.edu

September 16, 2019
"""
import torch
import torch.nn as nn
import operators as op
from aperture import generate_angular_spectrum_kernel
complex_mul = op.ComplexMul.apply


import numpy as np

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
    def __init__(self, shape, voxel_size, wavelength,  numerical_aperture=None, dtype=torch.float32, device=torch.device('cuda'),   **kwargs):
        super(MultislicePropagation, self).__init__()
        self.shape            = shape  
        self.voxel_size       = voxel_size
        self.distance_to_center = (self.shape[2]/2. - 1/2.) * self.voxel_size[2]
        self.propagate        = SingleSlicePropagation(self.shape[0:2], self.voxel_size[0],  wavelength, \
                                                       numerical_aperture=None, flag_band_limited=False, \
                                                       dtype=dtype, device=device)    
    def forward(self, obj, field_in=None):
        field = field_in
        if field is None:
            field = obj[:,:,0,:]
        else:
            field = complex_mul(field, obj[:,:,0,:])
        field = self.propagate(field, self.voxel_size[2])    
        for layer_idx in range(1, self.shape[2]):
            field = complex_mul(field, obj[:,:,layer_idx,:])
            if layer_idx < self.shape[2] - 1:
                #Propagate forward one layer
                field = self.propagate(field, self.voxel_size[2])
            else:
                field = self.propagate(field, -1. * self.distance_to_center)
        return field

class SingleSlicePropagation(nn.Module):
    '''
    Class for propagation for single slice
    '''
    def __init__(self, shape, pixel_size, wavelength, \
                 numerical_aperture=None,  flag_band_limited=False, \
                 dtype=torch.float32, device=torch.device('cuda')):
        super(SingleSlicePropagation, self).__init__()
        self.kernel_phase     = generate_angular_spectrum_kernel(shape, pixel_size, wavelength, \
                                                                 numerical_aperture=None,  flag_band_limited=False, \
                                                                 dtype=dtype, device=device)

    def forward(self, field_in, propagation_distance):
        if propagation_distance == 0:
            return field_in
        kernel = op.exp(abs(propagation_distance) * self.kernel_phase)
        kernel = kernel if propagation_distance > 0. else op.conj(kernel)
        field_out = op.convolve_kernel(field_in, kernel, n_dim=2, flag_inplace=False)
        return field_out
