"""
Utility functions for AET in Torch

David Ren      david.ren@berkeley.edu

October 7, 2019
"""

import operators as op
import numpy as np
import contexttimer

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

MAX_DIM = 512*512*512

def show3DStack(image_3d, axis = 2, cmap = "gray", clim = (0, 1), extent = (0, 1, 0, 1)):
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

def compare3DStack(stack_1, stack_2, axis = 2, cmap = "gray", clim = (0, 1), extent = (0, 1, 0, 1) , colorbar = True):
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
    _, ax        = plt.subplots(1, 2, figsize=(9, 5), sharex = 'all', sharey = 'all')
    plt.subplots_adjust(left=0.15, bottom=0.15)
    fig_1        = ax[0].imshow(image_1(current_idx), cmap = cmap,  clim = clim, extent = extent)
    ax[0].axis("off")
    ax[0].set_title("stack 1, layer: " + str(current_idx))
    plt.colorbar(fig_1, ax = ax[0])
    fig_2        = ax[1].imshow(image_2(current_idx), cmap = cmap,  clim = clim, extent = extent)
    ax[1].axis("off")
    ax[1].set_title("stack 2, layer: " + str(current_idx))
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
            self.coord_phase_1 = op.r2c(-2.0 * np.pi * self.kz * self.x)
            self.coord_phase_2 = op.r2c(-2.0 * np.pi * self.kx * self.z)
        elif self.axis == 1:
            self.coord_phase_1 = op.r2c(-2.0 * np.pi * self.kz * self.y)
            self.coord_phase_2 = op.r2c(-2.0 * np.pi * self.ky * self.z)
        elif self.axis == 2:
            self.coord_phase_1 = op.r2c(-2.0 * np.pi * self.kx * self.y)
            self.coord_phase_2 = op.r2c(-2.0 * np.pi * self.ky * self.x)

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
        if len(obj.shape) == 3:
            self.obj_rotate[self.range_crop_y, self.range_crop_x, self.range_crop_z,0] = obj 
        else:
            self.obj_rotate[self.range_crop_y, self.range_crop_x, self.range_crop_z,]  = obj         
        if self.axis == 0:
            self.obj_rotate = op.convolve_kernel(self.obj_rotate, shear_phase_1) #y,x,z
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([0,2,1,3]), shear_phase_2.permute([0,2,1,3])) #y,z,x
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([0,2,1,3]), shear_phase_1) #y,x,z

        elif self.axis == 1:
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([1,0,2,3]), shear_phase_1.permute([1,0,2,3])) #x,y,z
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([0,2,1,3]), shear_phase_2.permute([1,2,0,3])) #x,z,y
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([0,2,1,3]), shear_phase_1.permute([1,0,2,3])) #x,y,z
            self.obj_rotate = self.obj_rotate.permute([1,0,2,3])

        elif self.axis == 2:
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([2,0,1,3]), shear_phase_1.permute([2,0,1,3])) #z,y,x
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([0,2,1,3]), shear_phase_2.permute([2,1,0,3])) #z,x,y
            self.obj_rotate = op.convolve_kernel(self.obj_rotate.permute([0,2,1,3]), shear_phase_1.permute([2,0,1,3])) #z,y,x
            self.obj_rotate = self.obj_rotate.permute([1,2,0,3])
        if len(obj.shape) == 3:
            obj[:] = self.obj_rotate[self.range_crop_y, self.range_crop_x, self.range_crop_z,0]
        else:
            obj[:] = self.obj_rotate[self.range_crop_y, self.range_crop_x, self.range_crop_z,]
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

            shear_phase_1 = op.exp(op.multiply_complex(op._j, self.coord_phase_1 * alpha))
            shear_phase_2 = op.exp(op.multiply_complex(op._j, self.coord_phase_2 * beta))

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
                self.obj_rotate[...,0] = self.pad_value
                self.obj_rotate[...,1] = 0.0
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
            
            shear_phase_1 = op.exp(op.multiply_complex(op._j, self.coord_phase_1 * alpha))
            shear_phase_2 = op.exp(op.multiply_complex(op._j, self.coord_phase_2 * beta))

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

class ImageShiftGradientBased(nn.Module):
    """
    A class that solves for shift between measurement and prediction. This uses pytorch autograd, and is gradient based.
    """
    def __init__(self, shape, pixel_size, dtype=torch.float32, device=torch.device('cuda'),   **kwargs):
        super(ImageShiftGradientBased, self).__init__()
        self.ky_lin, self.kx_lin = generate_grid_2d(shape, pixel_size, flag_fourier=True, dtype=dtype, device=device)

    def forward(self, field, shift=None):   
        """
        Input parameters:
            - field: refocused field, before cropping
            - shift: estimated shift [y_shift, x_shift], default None (shift estimation off)
        """
        if shift is None:
            return field
        field_out = field.clone()
        for img_idx in range(field.shape[2]):
            y_shift = shift[0,img_idx]
            x_shift = shift[1,img_idx]
            kernel  = op.exp(op.multiply_complex(op._j, op.r2c(2 * np.pi * (self.kx_lin * x_shift + self.ky_lin * y_shift))))
            field_out[...,img_idx,:] = op.convolve_kernel(field[...,img_idx,:], kernel, n_dim=2)
        return field_out



class ImageShift:
    """
    A class that solves for shift between measurement and prediction. Several possible methods are implemented:
    a) "gradient" -- gradient method. Adjoint returns the updated shift in x and y (iterative)
                     to be used with gradient descent in solving for the object. Joint estimation using adjoint()
    b) "phase" -- pahse correlation method. Adjoint returns the fitted x and y shifts (single shot)
                     use solveShift()
    """
    def __init__(self, shape, pixel_size, verbose = False, shift_update_method = "phase", **kwargs):
        self.shape      = shape
        self.pixel_size = pixel_size
        self.verbose    = verbose

        #update methods
        self.update_method       = shift_update_method

        self.ky_lin, self.kx_lin = generate_grid_2d(self.shape[0:2], self.pixel_size, flag_fourier=True, **kwargs)
        
    def _shift_single(self, img_source, x_shift, y_shift):
        if x_shift==0 and y_shift==0:
            return img_source
        else:
            print(self.kx_lin.shape)
            kernel    = op.exp(op.multiply_complex(op._j, op.r2c(2 * np.pi * (self.kx_lin * x_shift + self.ky_lin * y_shift))))
            img_shift = op.convolve_kernel(op.r2c(img_source), kernel, n_dim=2)
        return img_shift[...,0]

    def forward(self, img_source, x_shift, y_shift):
        if len(img_source.shape) > 2:
            assert img_source.shape[2] == len(x_shift)
            assert img_source.shape[2] == len(y_shift)
            for idx in range(img_source.shape[2]):
                if x_shift[idx] == 0 and y_shift[idx] == 0:
                    continue
                else:
                    img_source[:,:,idx] = self._shift_single(img_source[:,:,idx], x_shift[idx], y_shift[idx])
        else:
            if x_shift == 0 and y_shift == 0:
                pass
            else:
                img_source = self._shift_single(img_source, x_shift, y_shift)
        return img_source
            
    def solve_shift(self, img_predict, img_measure, x_init = 0.0, y_init = 0.0, shift_bound = 10.0):
        if self.update_method == "phase":
            if len(img_measure.shape) > 2:
                y_shift = np.zeros(img_measure.shape[2])
                x_shift = np.zeros(img_measure.shape[2])

                for idx in range(img_measure.shape[2]):            
                    a_f = torch.fft(op.r2c(img_predict[:,:,idx]), signal_ndim=2)
                    b_f_conj = op.conj(torch.fft(op.r2c(img_measure[:,:,idx]), signal_ndim=2))
                    prod_f = op.multiply_complex(a_f, b_f_conj)
                    corr = op.fftshift(op.abs(torch.ifft(op.exp(op.multiply_complex(op._j, op.r2c(op.angle(prod_f)))),signal_ndim=2)), axes=[0,1])
                    (y_shift_temp, x_shift_temp) = np.unravel_index(torch.argmax(corr).cpu().detach(), corr.shape)
                    y_shift[idx] = (y_shift_temp-corr.shape[0]//2)  * self.pixel_size
                    x_shift[idx] = (x_shift_temp-corr.shape[1]//2)  * self.pixel_size
            else:                
                a_f = torch.fft(op.r2c(img_predict), signal_ndim=2)
                b_f_conj = op.conj(torch.fft(op.r2c(img_measure), signal_ndim=2))
                prod_f = op.multiply_complex(a_f, b_f_conj)
                corr = op.fftshift(op.abs(torch.ifft(op.exp(op.multiply_complex(op._j, op.r2c(op.angle(prod_f)))),signal_ndim=2)), axes=[0,1])
                (y_shift_temp, x_shift_temp) = np.unravel_index(torch.argmax(corr).cpu().detach(), corr.shape)
                y_shift = (y_shift_temp-corr.shape[0]//2)  * self.pixel_size
                x_shift = (x_shift_temp-corr.shape[1]//2)  * self.pixel_size
                
        elif self.update_method == "cross":
            if len(img_measure.shape) > 2:
                y_shift = np.zeros(img_measure.shape[2])
                x_shift = np.zeros(img_measure.shape[2])

                for idx in range(img_measure.shape[2]):            
                    a_f = torch.fft(op.r2c(img_predict[:,:,idx]), signal_ndim=2)
                    b_f_conj = op.conj(torch.fft(op.r2c(img_measure[:,:,idx]), signal_ndim=2))
                    prod_f = op.multiply_complex(a_f, b_f_conj)
                    corr = op.fftshift(op.abs(torch.ifft(prod_f,signal_ndim=2)), axes[0,1])
                    (y_shift_temp, x_shift_temp) = np.unravel_index(torch.argmax(corr).cpu().detach(), corr.shape)
                    y_shift[idx] = (y_shift_temp-corr.shape[0]//2)  * self.pixel_size
                    x_shift[idx] = (x_shift_temp-corr.shape[1]//2)  * self.pixel_size
            else:                
                a_f = torch.fft(op.r2c(img_predict), signal_ndim=2)
                b_f_conj = op.conj(torch.fft(op.r2c(img_measure), signal_ndim=2))
                prod_f = op.multiply_complex(a_f, b_f_conj)
                corr = op.fftshift(op.abs(torch.ifft(prod_f,signal_ndim=2)), axes[0,1])
                (y_shift_temp, x_shift_temp) = np.unravel_index(torch.argmax(corr).cpu().detach(), corr.shape)
                y_shift = (y_shift_temp-corr.shape[0]//2)  * self.pixel_size
                x_shift = (x_shift_temp-corr.shape[1]//2)  * self.pixel_size
                
        if (np.abs(x_shift) > shift_bound).any() or (np.abs(y_shift) > shift_bound).any():
            if self.verbose:
                print("WARNING: Shift estimation diverged! Not updating! ")
                print("y_shift", y_shift)
                print("x_shift", x_shift)
                y_shift -= y_shift
                x_shift -= x_shift
                
        return x_shift, y_shift

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
        assert len(obj_in.shape)  == 4
        ctx.factor = factor
        if factor == 1:
            return obj_in
        n_y, n_x, n_z, n_c = obj_in.shape
        obj_out = obj_in.reshape(n_y, n_x, n_z//factor, factor, n_c).sum(3)
        return obj_out

    @staticmethod
    def backward(ctx, grad_output):
        factor = ctx.factor
        if factor == 1:
            return grad_output, None

        return grad_output.repeat_interleave(factor, dim=-2), None
