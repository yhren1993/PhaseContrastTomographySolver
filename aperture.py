"""
Aperture functions for AET in Torch

David Ren      david.ren@berkeley.edu

September 16, 2019
"""

import operators as op
import utilities as util
import numpy as np

import torch
import torch.nn as nn

def generate_hard_pupil(shape, pixel_size, numerical_aperture, wavelength, \
                   dtype=torch.float32, device=torch.device('cuda')):
    """
    This function generates pupil function(circular function) given shape, pixel_size, na, and wavelength
    """
    assert len(shape) == 2, "pupil should be two dimensional!"
    ky_lin, kx_lin = util.generate_grid_2d(shape, pixel_size, flag_fourier=True, dtype=dtype, device=device)

    pupil_radius = numerical_aperture/wavelength
    pupil        = (kx_lin**2 + ky_lin**2 <= pupil_radius**2).type(dtype)
    return op.r2c(pupil)

def generate_angular_spectrum_kernel(shape, pixel_size, wavelength, \
                                     numerical_aperture=None,  flag_band_limited=True, \
                                     dtype=torch.float32, device=torch.device('cuda')):
    """
    Function that generates angular spectrum propagation kernel WITHOUT the distance
    The angular spectrum has the following form:
    p = exp(distance * kernel)
    kernel = 1j * 2 * pi * sqrt((ri/wavelength)**2-x**2-y**2)
    and this function generates the kernel only!
    """
    assert len(shape) == 2, "pupil should be two dimensional!"
    ky_lin, kx_lin = util.generate_grid_2d(shape, pixel_size, flag_fourier=True, dtype=dtype, device=device)
    if flag_band_limited:
        assert numerical_aperture is not None, "need to provide numerical aperture of the system!"
        pupil_crop    = op.r2c(generate_hard_pupil(shape, pixel_size, numerical_aperture, wavelength))
    else: 
        pupil_crop    = 1.0
    prop_kernel = 2.0 * np.pi * pupil_crop * \
                  op.exponentiate(op.r2c((1./wavelength)**2 - kx_lin**2 - ky_lin**2), 0.5)
    return op.multiply_complex(op._j, prop_kernel)

class Pupil(nn.Module):
    """
    Class for applying pupil in forward model computation
    """
    def __init__(self, shape, pixel_size, wavelength, \
                 numerical_aperture = 1.0, pupil = None, \
                 dtype=torch.float32, device=torch.device('cuda'), **kwargs):
        super(Pupil, self).__init__()
        if pupil is not None:
            self.pupil = pupil.type(dtype).to(device)
            if len(self.pupil.shape) == 2:
                self.pupil = op.r2c(self.pupil)
        else:
            self.pupil = generate_hard_pupil(shape, pixel_size, numerical_aperture, wavelength, dtype, device)
    def get_pupil(self):
        return self.pupil.cpu()
    def forward(self, field):
        field_out = op.convolve_kernel(field, self.pupil, n_dim=2, flag_inplace=False)
        return field_out