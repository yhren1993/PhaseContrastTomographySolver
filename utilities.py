"""
Utility functions for AET in Torch

David Ren      david.ren@berkeley.edu

October 7, 2019
"""

import operators as op
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

MAX_DIM = 512*512*512

def show3DStack(image_3d, axis = 2, cmap = "gray", clim = None, extent = (0, 1, 0, 1)):
    if clim is None:
        clim = (np.min(image_3d), np.max(image_3d)) 
    if axis == 0:
        image  = lambda index: image_3d[index, :, :]
    elif axis == 1:
        image  = lambda index: image_3d[:, index, :]
    else:
        image  = lambda index: image_3d[:, :, index]

    current_idx= 0
    _, ax      = plt.subplots(1, 1, figsize=(6.5, 5))
    plt.subplots_adjust(left=0.15, bottom=0.15)
    fig        = ax.imshow(image(current_idx), cmap = cmap,  clim = clim, extent = extent)
    ax.set_title("layer: " + str(current_idx))
    plt.colorbar(fig, ax=ax)
    plt.axis('off')
    ax_slider  = plt.axes([0.15, 0.1, 0.65, 0.03])
    slider_obj = Slider(ax_slider, "layer", 0, image_3d.shape[axis]-1, valinit=current_idx, valfmt='%d')
    def update_image(index):
        global current_idx
        index       = int(index)
        current_idx = index
        ax.set_title("layer: " + str(index))
        fig.set_data(image(index))
    def arrow_key(event):
        global current_idx
        if event.key == "left":
            if current_idx-1 >=0:
                current_idx -= 1
        elif event.key == "right":
            if current_idx+1 < image_3d.shape[axis]:
                current_idx += 1
        slider_obj.set_val(current_idx)
    slider_obj.on_changed(update_image)
    plt.gcf().canvas.mpl_connect("key_release_event", arrow_key)
    plt.show()
    return slider_obj

def compare3DStack(stack_1, stack_2, axis = 2, cmap = "gray", clim = (0, 1), extent = (0, 1, 0, 1) , colorbar = True, flag_vertical = False):
    assert stack_1.shape == stack_2.shape, "shape of two input stacks should be the same!"

    if axis == 0:
        image_1  = lambda index: stack_1[index, :, :]
        image_2  = lambda index: stack_2[index, :, :]
    elif axis == 1:
        image_1  = lambda index: stack_1[:, index, :]
        image_2  = lambda index: stack_2[:, index, :]
    else:
        image_1  = lambda index: stack_1[:, :, index]
        image_2  = lambda index: stack_2[:, :, index]

    current_idx  = 0
    if flag_vertical:
        _, ax        = plt.subplots(2, 1, figsize=(10, 2.5), sharex = 'all', sharey = 'all')
    else:
        _, ax        = plt.subplots(1, 2, figsize=(9, 5), sharex = 'all', sharey = 'all')
    plt.subplots_adjust(left=0.15, bottom=0.15)
    fig_1        = ax[0].imshow(image_1(current_idx), cmap = cmap,  clim = clim, extent = extent)
    ax[0].axis("off")
    ax[0].set_title("stack 1, layer: " + str(current_idx))
    if colorbar:
        plt.colorbar(fig_1, ax = ax[0])
    fig_2        = ax[1].imshow(image_2(current_idx), cmap = cmap,  clim = clim, extent = extent)
    ax[1].axis("off")
    ax[1].set_title("stack 2, layer: " + str(current_idx))
    if colorbar:
        plt.colorbar(fig_2, ax = ax[1])
    ax_slider    = plt.axes([0.10, 0.05, 0.65, 0.03])
    slider_obj   = Slider(ax_slider, 'layer', 0, stack_1.shape[axis]-1, valinit=current_idx, valfmt='%d')
    def update_image(index):
        global current_idx
        index       = int(index)
        current_idx = index
        ax[0].set_title("stack 1, layer: " + str(index))
        fig_1.set_data(image_1(index))
        ax[1].set_title("stack 2, layer: " + str(index))
        fig_2.set_data(image_2(index))
    def arrow_key(event):
        global current_idx
        if event.key == "left":
            if current_idx-1 >=0:
                current_idx -= 1
        elif event.key == "right":
            if current_idx+1 < stack_1.shape[axis]:
                current_idx += 1
        slider_obj.set_val(current_idx)
    slider_obj.on_changed(update_image)
    plt.gcf().canvas.mpl_connect("key_release_event", arrow_key)
    plt.show()
    return slider_obj

def compare4DStack(stack_1, stack_2, cmap = "gray", clim = (0, 1), extent = (0, 1, 0, 1) , colorbar = True, flag_vertical = False):
    assert stack_1.shape == stack_2.shape, "shape of two input stacks should be the same!"
    axis, axis2 = 2, 3
    image_1  = lambda index: stack_1[:, :, index[0], index[1]]
    image_2  = lambda index: stack_2[:, :, index[0], index[1]]

    current_idx1, current_idx2  = 0, 0
    if flag_vertical:
        _, ax        = plt.subplots(2, 1, figsize=(10, 2.5), sharex = 'all', sharey = 'all')
    else:
        _, ax        = plt.subplots(1, 2, figsize=(9, 5), sharex = 'all', sharey = 'all')
    plt.subplots_adjust(left=0.15, bottom=0.15)
    fig_1        = ax[0].imshow(image_1((current_idx1,current_idx2)), cmap = cmap,  clim = clim, extent = extent)
    ax[0].axis("off")
    ax[0].set_title("stack 1, layer: " + str(current_idx1) + " and " + str(current_idx2))
    if colorbar:
        plt.colorbar(fig_1, ax = ax[0])
    fig_2        = ax[1].imshow(image_2((current_idx1,current_idx2)), cmap = cmap,  clim = clim, extent = extent)
    ax[1].axis("off")
    ax[1].set_title("stack 2, layer: " + str(current_idx1) + " and " + str(current_idx2))
    if colorbar:
        plt.colorbar(fig_2, ax = ax[1])
    ax_slider    = plt.axes([0.10, 0.10, 0.65, 0.03])
    slider_obj   = Slider(ax_slider, 'layer', 0, stack_1.shape[axis]-1, valinit=current_idx1, valfmt='%d')
    def update_image(index):
        global current_idx1
        global current_idx2
        index       = int(index)
        current_idx1 = index
        current_idx2 = current_idx2
        ax[0].set_title("stack 1, layer: " + str(current_idx1) + " and " + str(current_idx2))
        fig_1.set_data(image_1((current_idx1,current_idx2)))
        ax[1].set_title("stack 2, layer: " + str(current_idx1) + " and " + str(current_idx2))
        fig_2.set_data(image_2((current_idx1,current_idx2)))
    ax_slider2    = plt.axes([0.10, 0.05, 0.65, 0.03])
    slider_obj2   = Slider(ax_slider2, 'layer', 0, stack_1.shape[axis2]-1, valinit=current_idx2, valfmt='%d')
    def update_image2(index):
        global current_idx1
        global current_idx2
        index       = int(index)
        current_idx1= current_idx1
        current_idx2= index
        ax[0].set_title("stack 1, layer: " + str(current_idx1) + " and " + str(current_idx2))
        fig_1.set_data(image_1((current_idx1,current_idx2)))
        ax[1].set_title("stack 2, layer: " + str(current_idx1) + " and " + str(current_idx2))
        fig_2.set_data(image_2((current_idx1,current_idx2)))        
    def arrow_key(event):
        global current_idx1
        global current_idx2
        current_idx2 = current_idx2
        if event.key == "left":
            if current_idx1-1 >=0:
                current_idx1 -= 1
        elif event.key == "right":
            if current_idx1+1 < stack_1.shape[axis]:
                current_idx1 += 1
        slider_obj.set_val(current_idx1)
    slider_obj.on_changed(update_image)
    slider_obj2.on_changed(update_image2)
    plt.gcf().canvas.mpl_connect("key_release_event", arrow_key)
    plt.show()
    return slider_obj, slider_obj2


def generate_grid_1d(shape, pixel_size = 1, flag_fourier = False, dtype = torch.float32, device = torch.device('cuda')):
    """
    This function generates 1D Fourier grid, and is centered at the middle of the array
    Inputs:
        shape    - length of the array
        pixel_size      - pixel size
    Optional parameters:
        flag_fourier - flag indicating whether the final array is circularly shifted
                     should be false when computing real space coordinates
                     should be true when computing Fourier coordinates
    Outputs:
        x_lin       - 1D grid (real or fourier)

    """
    pixel_size = 1./pixel_size/shape if flag_fourier else pixel_size
    x_lin = (torch.arange(shape, dtype=dtype, device=device) - shape//2) * pixel_size
    if flag_fourier:
        x_lin = torch.roll(x_lin, -1 * int(shape)//2)
    return x_lin

def generate_grid_2d(shape, pixel_size = 1, flag_fourier = False, dtype = torch.float32, device = torch.device('cuda')):
    """
    This function generates 2D Fourier grid, and is centered at the middle of the array
    Inputs:
        shape              - shape of the grid (number of y pixels, number of x pixels)
        pixel_size         - pixel size
    Optional parameters:
        flag_fourier       - flag indicating whether the final array is circularly shifted
                             should be false when computing real space coordinates
                             should be true when computing Fourier coordinates
    Outputs:
        y_lin, x_lin       - 2D grid
    Usage:
        y_lin, x_lin = generate_grid_2d(...)

    """    
    assert len(shape) == 2, "shape should be two dimensional!"
    #recompute pixel size for fourier space sampling
    y_lin  = generate_grid_1d(shape[0], pixel_size, flag_fourier = flag_fourier, dtype=dtype, device=device)
    x_lin  = generate_grid_1d(shape[1], pixel_size, flag_fourier = flag_fourier, dtype=dtype, device=device)
    y_lin, x_lin = torch.meshgrid(y_lin, x_lin)
    return y_lin, x_lin


class ImageRotation:
    """
    A rotation class compute 3D rotation using FFT
    """
    def __init__(self, shape, axis = 0, pad = True, pad_value = 0, dtype = torch.float32, device = torch.device('cuda')):
        self.dim       = np.array(shape)
        self.axis      = axis
        self.pad_value = pad_value
        if pad:
            self.pad_size            = np.ceil(self.dim / 2.0).astype('int')
            self.pad_size[self.axis] = 0
            self.dim                += 2*self.pad_size
        else:
            self.pad_size  = np.asarray([0,0,0])
        
        self.dim          = [int(size) for size in self.dim]

        self.range_crop_y = slice(self.pad_size[0],self.pad_size[0] + shape[0])
        self.range_crop_x = slice(self.pad_size[1],self.pad_size[1] + shape[1])
        self.range_crop_z = slice(self.pad_size[2],self.pad_size[2] + shape[2])

        self.y            = generate_grid_1d(self.dim[0], dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        self.x            = generate_grid_1d(self.dim[1], dtype=dtype).unsqueeze(0).unsqueeze(-1)
        self.z            = generate_grid_1d(self.dim[2], dtype=dtype).unsqueeze(0).unsqueeze(0)
        
        self.ky           = generate_grid_1d(self.dim[0], flag_fourier = True, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        self.kx           = generate_grid_1d(self.dim[1], flag_fourier = True, dtype=dtype).unsqueeze(0).unsqueeze(-1)
        self.kz           = generate_grid_1d(self.dim[2], flag_fourier = True, dtype=dtype).unsqueeze(0).unsqueeze(0)

        #Compute FFTs sequentially if object size is too large
        self.slice_per_tile = int(np.min([np.floor(MAX_DIM * self.dim[self.axis] / np.prod(self.dim)), self.dim[self.axis]]))            
        self.dtype          = dtype
        self.device         = device

        if self.axis == 0:
            self.coord_phase_1 = -2.0 * np.pi * self.kz * self.x
            self.coord_phase_2 = -2.0 * np.pi * self.kx * self.z
        elif self.axis == 1:
            self.coord_phase_1 = -2.0 * np.pi * self.kz * self.y
            self.coord_phase_2 = -2.0 * np.pi * self.ky * self.z
        elif self.axis == 2:
            self.coord_phase_1 = -2.0 * np.pi * self.kx * self.y
            self.coord_phase_2 = -2.0 * np.pi * self.ky * self.x

    def _rotate_3d(self, obj, shear_phase_1, shear_phase_2):
        """
        This function rotates a 3D image by shearing, (applied in Fourier space)
        ** Note: the rotation is performed along the z axis

        [ cos(theta)  -sin(theta) ] = [ 1  alpha ] * [ 1     0  ] * [ 1  alpha ]
        [ sin(theta)  cos(theta)  ]   [ 0    1   ]   [ beta  1  ]   [ 0    1   ]
        alpha = tan(theta/2)
        beta = -sin(theta)

        Shearing in one shapeension is applying phase shift in 1D fourier transform
        Input:
          obj: 3D array (supposed to be an image), the axes are [z,y,x]
          theta: desired angle of rotation in *degrees*
        Output:
          obj_rotate: rotate 3D array
        """
        flag_complex = obj.is_complex()
        self.obj_rotate[self.range_crop_y, self.range_crop_x, self.range_crop_z] = op.r2c(obj)
        if self.axis == 0:
            self.obj_rotate = op.convolve_kernel(self.obj_rotate, shear_phase_1) #y,x,z
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([0,2,1]), shear_phase_2.permute([0,2,1])) #y,z,x
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([0,2,1]), shear_phase_1) #y,x,z

        elif self.axis == 1:
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([1,0,2]), shear_phase_1.permute([1,0,2])) #x,y,z
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([0,2,1]), shear_phase_2.permute([1,2,0])) #x,z,y
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([0,2,1]), shear_phase_1.permute([1,0,2])) #x,y,z
            self.obj_rotate = self.obj_rotate.permute([1,0,2])

        elif self.axis == 2:
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([2,0,1]), shear_phase_1.permute([2,0,1])) #z,y,x
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([0,2,1]), shear_phase_2.permute([2,1,0])) #z,x,y
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([0,2,1]), shear_phase_1.permute([2,0,1])) #z,y,x
            self.obj_rotate = self.obj_rotate.permute([1,2,0])
        if flag_complex:
            obj[:] = self.obj_rotate[self.range_crop_y, self.range_crop_x, self.range_crop_z]
        else:
            obj[:] = self.obj_rotate[self.range_crop_y, self.range_crop_x, self.range_crop_z].real
        return obj

    def forward(self, obj, theta):
        self.theta = theta
        if theta == 0:
            return obj
        else:
            flag_cpu = False
            if not obj.is_cuda:
                flag_cpu = True
                obj = obj.cuda()
            theta      *= np.pi / 180.0
            alpha       = 1.0 * np.tan(theta / 2.0)
            beta        = np.sin(-1.0 * theta)

            shear_phase_1 = torch.exp(1j * self.coord_phase_1 * alpha)
            shear_phase_2 = torch.exp(1j * self.coord_phase_2 * beta)

            self.dim[self.axis] = self.slice_per_tile
            self.obj_rotate = op.r2c(torch.ones([self.dim[0], self.dim[1], self.dim[2]], dtype=self.dtype, device=self.device) * self.pad_value)

            for idx_start in range(0, obj.shape[self.axis], self.slice_per_tile):
                idx_end = np.min([obj.shape[self.axis], idx_start+self.slice_per_tile])
                idx_slice = slice(idx_start, idx_end)
                self.dim[self.axis] = int(idx_end - idx_start)
                if self.axis == 0:
                    self.range_crop_y = slice(0, self.dim[self.axis])
                    obj[idx_slice,:,:] = self._rotate_3d(obj[idx_slice,:,:], shear_phase_1, shear_phase_2)
                elif self.axis == 1:
                    self.range_crop_x = slice(0, self.dim[self.axis])
                    obj[:,idx_slice,:] = self._rotate_3d(obj[:,idx_slice,:], shear_phase_1, shear_phase_2)
                elif self.axis == 2:
                    self.range_crop_z = slice(0, self.dim[self.axis])
                    obj[:,:,idx_slice] = self._rotate_3d(obj[:,:,idx_slice], shear_phase_1, shear_phase_2)
                self.obj_rotate[:] = self.pad_value + 0.j
            self.dim[self.axis] = obj.shape[self.axis]
            self.obj_rotate = None
            if flag_cpu:
                obj = obj.cpu()
            return obj

    def backward(self, obj):
        theta = -1 * self.theta
        if theta == 0:
            return obj
        else:
            if not obj.is_cuda:
                obj = obj.cuda()            
            theta      *= np.pi / 180.0
            alpha       = 1.0 * np.tan(theta / 2.0)
            beta        = np.sin(-1.0 * theta)
            
            shear_phase_1 = torch.exp(1j * self.coord_phase_1 * alpha)
            shear_phase_2 = torch.exp(1j * self.coord_phase_2 * beta)

            self.dim[self.axis] = self.slice_per_tile
            self.obj_rotate = op.r2c(torch.zeros([self.dim[0], self.dim[1], self.dim[2]], dtype=self.dtype, device=self.device))

            for idx_start in range(0, obj.shape[self.axis], self.slice_per_tile):
                idx_end = np.min([obj.shape[self.axis], idx_start+self.slice_per_tile])
                idx_slice = slice(idx_start, idx_end)
                self.dim[self.axis] = int(idx_end - idx_start)
                if self.axis == 0:
                    self.range_crop_y = slice(0, self.dim[self.axis])
                    obj[idx_slice,:,:] = self._rotate_3d(obj[idx_slice,:,:], alpha, beta, shear_phase_1, shear_phase_2)
                elif self.axis == 1:
                    self.range_crop_x = slice(0, self.dim[self.axis])
                    obj[:,idx_slice,:] = self._rotate_3d(obj[:,idx_slice,:], alpha, beta, shear_phase_1, shear_phase_2)
                elif self.axis == 2:
                    self.range_crop_z = slice(0, self.dim[self.axis])
                    obj[:,:,idx_slice] = self._rotate_3d(obj[:,:,idx_slice], alpha, beta, shear_phase_1, shear_phase_2)
                self.obj_rotate[:] = 0.0
            self.dim[self.axis] = obj.shape[self.axis]
            self.obj_rotate = None         
            if not obj.is_cuda:
                obj = obj.cpu()
            return obj

class BinObject(torch.autograd.Function):
    '''
    Class that bins the object along the direction of beam propagation (z)
    inputs:
    obj_in: input object 
    factor: factor at which the object will be binned
    '''
    @staticmethod
    def forward(ctx, obj_in, factor):
        assert (obj_in.shape[2] % factor) == 0
        assert len(obj_in.shape) == 3
        ctx.factor = factor
        if factor == 1:
            return obj_in
        n_y, n_x, n_z = obj_in.shape
        obj_out = obj_in.reshape(n_y, n_x, n_z//factor, factor).sum(3)
        return obj_out

    @staticmethod
    def backward(ctx, grad_output):
        factor = ctx.factor
        if factor == 1:
            return grad_output, None

        return grad_output.repeat_interleave(factor, dim=-1), None


