"""
Complex operators computation for Torch

Michael Chen   mchen0405@berkeley.edu
David Ren      david.ren@berkeley.edu

September 16, 2019
"""

import torch
import numpy as np


def r2c(real_tensor):
    '''Convert from real to complex'''
    return real_tensor+0j

def convolve_kernel(tensor_in, kernel, n_dim=1, flag_inplace=True):
    '''
    Compute convolution FFT(tensor_in) and kernel
    Required Args:
        tensor_in: variable 1 in real space
        kernel: variable 2 in reciprocal space

    Optional Args [default]
        n_dim: number of dimensions to compute convolution [1]
        flag_inplace: Whether or not compute convolution inplace, result saved in 'tensor_in' [True]
    '''
    dim = [-n_dim+i for i in range(n_dim)]
    if flag_inplace:
        tensor_in  = torch.fft.fftn(tensor_in, dim=dim)
        tensor_in *= kernel
        tensor_in  = torch.fft.ifftn(tensor_in, dim=dim)
        return tensor_in
    else:
        output  = torch.fft.fftn(tensor_in, dim=dim)
        output *= kernel
        output  = torch.fft.ifftn(output, dim=dim)
        return output

def fftshift(tensor_in, axes=None):
    '''Custom implemented fftshift operator'''
    ret = tensor_in.clone()
    axes= np.atleast_1d(axes)
    for axis in axes:
        ret =  torch.roll(ret, ret.shape[axis]//2, dims=int(axis))
    return ret

def ifftshift(tensor_in, axes=None):
    '''Custom implemented ifftshift operator'''
    ret = tensor_in.clone()
    axes= np.atleast_1d(axes)
    for axis in axes:
        ret =  torch.roll(ret, -1*int(ret.shape[axis]//2), dims=int(axis))
    return ret

class ComplexAbs(torch.autograd.Function):
    '''Absolute value class for autograd'''
    @staticmethod
    def forward(ctx, tensor_in):
        output = torch.abs(tensor_in)
        ctx.save_for_backward(torch.angle(tensor_in))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tensor_in_angle,     = ctx.saved_tensors
        return 0.5*torch.exp(1j * tensor_in_angle) * grad_output