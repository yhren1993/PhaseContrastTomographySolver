"""
Shift functions for Tomography in Torch & Numpy
Gradient based shift is implemented as a nn.Module class in Torch.
Correlation based shift is using part of the code in skimage, implemented in numpy, and can be found at:
https://github.com/scikit-image/scikit-image

David Ren      david.ren@berkeley.edu

April 23, 2020
"""
import utilities as util
import operators as op

import torch
import torch.nn as nn

import numpy as np
import numpy.fft as fft

possible_methods = [
                    "gradient",\
                    "phase_correlation",\
                    "cross_correlation",\
                    "hybrid_correlation"
                   ]
correlation_methods = [
                    "phase_correlation",\
                    "cross_correlation",\
                    "hybrid_correlation"
                   ]                    
def is_correlation_method(method):
    return method in correlation_methods

def is_valid_method(method):
    return method in possible_methods

class ImageShiftCorrelationBased():
    """
    Class written to register stack of images for AET.
    Uses correlation based method to determine subpixel shift between predicted and measured images.
    Input parameters:
        - shape: shape of the image
        - pixel_size: pixel size of the image
        - upsample_factor: precision of shift algorithm, to 1/upsample_factor accuracy.
    """    
    def __init__(self, shape, upsample_factor=10, method="cross_correlation", dtype=torch.float32, device=torch.device('cuda')):
        pixel_size = 1.0
        self.ky_lin, self.kx_lin = util.generate_grid_2d(shape, pixel_size, flag_fourier=True, dtype=dtype, device=device)
        self.upsample_factor = upsample_factor
        self.method = method
    def _upsampled_dft(self, data, upsampled_region_size,
                       upsample_factor=1, axis_offsets=None):
        """
        Upsampled DFT by matrix multiplication.

        This code is intended to provide the same result as if the following
        operations were performed:
            - Embed the array "data" in an array that is ``upsample_factor`` times
              larger in each dimension.  ifftshift to bring the center of the
              image to (1,1).
            - Take the FFT of the larger array.
            - Extract an ``[upsampled_region_size]`` region of the result, starting
              with the ``[axis_offsets+1]`` element.

        It achieves this result by computing the DFT in the output array without
        the need to zeropad. Much faster and memory efficient than the zero-padded
        FFT approach if ``upsampled_region_size`` is much smaller than
        ``data.size * upsample_factor``.

        Parameters
        ----------
        data : array
            The input data array (DFT of original data) to upsample.
        upsampled_region_size : integer or tuple of integers, optional
            The size of the region to be sampled.  If one integer is provided, it
            is duplicated up to the dimensionality of ``data``.
        upsample_factor : integer, optional
            The upsampling factor.  Defaults to 1.
        axis_offsets : tuple of integers, optional
            The offsets of the region to be sampled.  Defaults to None (uses
            image center)

        Returns
        -------
        output : ndarray
                The upsampled DFT of the specified region.
        """
        # if people pass in an integer, expand it to a list of equal-sized sections
        if not hasattr(upsampled_region_size, "__iter__"):
            upsampled_region_size = [upsampled_region_size, ] * data.ndim
        else:
            if len(upsampled_region_size) != data.ndim:
                raise ValueError("shape of upsampled region sizes must be equal "
                                 "to input data's number of dimensions.")

        if axis_offsets is None:
            axis_offsets = [0, ] * data.ndim
        else:
            if len(axis_offsets) != data.ndim:
                raise ValueError("number of axis offsets must be equal to input "
                                 "data's number of dimensions.")

        im2pi = 1j * 2 * np.pi

        dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

        for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
            kernel = ((np.arange(ups_size) - ax_offset)[:, None]
                      * fft.fftfreq(n_items, upsample_factor))
            kernel = np.exp(-im2pi * kernel)

            # Equivalent to:
            #   data[i, j, k] = kernel[i, :] @ data[j, k].T
            data = np.tensordot(kernel, data, axes=(1, -1))
        return data

    def _compute_error(self, cross_correlation_max, src_amp, target_amp):
        """
        Compute RMS error metric between ``src_image`` and ``target_image``.

        Parameters
        ----------
        cross_correlation_max : complex
            The complex value of the cross correlation at its maximum point.
        src_amp : float
            The normalized average image intensity of the source image
        target_amp : float
            The normalized average image intensity of the target image
        """
        error = 1.0 - cross_correlation_max * cross_correlation_max.conj() /\
            (src_amp * target_amp)
        return np.sqrt(np.abs(error))


    def _cross_correlation(self, reference_image, moving_image, upsample_factor=1,
                          method = "cross_correlation", space="real", return_error=True):
        """Efficient subpixel image translation registration by cross-correlation.

        This code gives the same precision as the FFT upsampled cross-correlation
        in a fraction of the computation time and with reduced memory requirements.
        It obtains an initial estimate of the cross-correlation peak by an FFT and
        then refines the shift estimation by upsampling the DFT only in a small
        neighborhood of that estimate by means of a matrix-multiply DFT.

        Parameters
        ----------
        reference_image : array
            Reference image.
        moving_image : array
            Image to register. Must be same dimensionality as
            ``reference_image``.
        upsample_factor : int, optional
            Upsampling factor. Images will be registered to within
            ``1 / upsample_factor`` of a pixel. For example
            ``upsample_factor == 20`` means the images will be registered
            within 1/20th of a pixel. Default is 1 (no upsampling)
        method: string, one of "cross_correlation", "phase_correlation", or "hybrid_correlation"
        space : string, one of "real" or "fourier", optional
            Defines how the algorithm interprets input data. "real" means data
            will be FFT'd to compute the correlation, while "fourier" data will
            bypass FFT of input data. Case insensitive.
        return_error : bool, optional
            Returns error and phase difference if on, otherwise only
            shifts are returned

        Returns
        -------
        shifts : ndarray
            Shift vector (in pixels) required to register ``moving_image``
            with ``reference_image``. Axis ordering is consistent with
            numpy (e.g. Z, Y, X)
        error : float
            Translation invariant normalized RMS error between
            ``reference_image`` and ``moving_image``.
        phasediff : float
            Global phase difference between the two images (should be
            zero if images are non-negative).

        References
        ----------
        .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
               "Efficient subpixel image registration algorithms,"
               Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`
        .. [2] James R. Fienup, "Invariant error metrics for image reconstruction"
               Optics Letters 36, 8352-8357 (1997). :DOI:`10.1364/AO.36.008352`

        """
        # images must be the same shape
        if reference_image.shape != moving_image.shape:
            raise ValueError("images must be same shape")

        # assume complex data is already in Fourier space
        if space.lower() == 'fourier':
            src_freq = reference_image
            target_freq = moving_image
        # real data needs to be fft'd.
        elif space.lower() == 'real':
            src_freq = fft.fftn(reference_image)
            target_freq = fft.fftn(moving_image)
        else:
            raise ValueError('space argument must be "real" or "fourier"')
        # Whole-pixel shift - Compute cross-correlation by an IFFT
        shape = src_freq.shape
        image_product = src_freq * target_freq.conj()
        if method == "phase_correlation":
            image_product = np.exp(1.0j*np.angle(image_product))
        elif method == "hybrid_correlation":
            image_product = np.sqrt(np.abs(image_product))*np.exp(1.0j*np.angle(image_product))
        else:
            raise ValueError('method argument not valid.')
        cross_correlation = fft.ifftn(image_product)

        # Locate maximum
        maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                                  cross_correlation.shape)
        midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

        shifts = np.array(maxima, dtype=np.float64)
        shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

        if upsample_factor == 1:
            if return_error:
                src_amp = np.sum(np.abs(src_freq) ** 2) / src_freq.size
                target_amp = np.sum(np.abs(target_freq) ** 2) / target_freq.size
                CCmax = cross_correlation[maxima]
        # If upsampling > 1, then refine estimate with matrix multiply DFT
        else:
            # Initial shift estimate in upsampled grid
            shifts = np.round(shifts * upsample_factor) / upsample_factor
            upsampled_region_size = np.ceil(upsample_factor * 1.5)
            # Center of output array at dftshift + 1
            dftshift = np.fix(upsampled_region_size / 2.0)
            upsample_factor = np.array(upsample_factor, dtype=np.float64)
            normalization = (src_freq.size * upsample_factor ** 2)
            # Matrix multiply DFT around the current shift estimate
            sample_region_offset = dftshift - shifts*upsample_factor
            cross_correlation = self._upsampled_dft(image_product.conj(),
                                               upsampled_region_size,
                                               upsample_factor,
                                               sample_region_offset).conj()
            cross_correlation /= normalization
            # Locate maximum and map back to original pixel grid
            maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                                      cross_correlation.shape)
            CCmax = cross_correlation[maxima]

            maxima = np.array(maxima, dtype=np.float64) - dftshift

            shifts = shifts + maxima / upsample_factor

            if return_error:
                src_amp = self._upsampled_dft(src_freq * src_freq.conj(),
                                         1, upsample_factor)[0, 0]
                src_amp /= normalization
                target_amp = self._upsampled_dft(target_freq * target_freq.conj(),
                                            1, upsample_factor)[0, 0]
                target_amp /= normalization

        # If its only one row or column the shift along that dimension has no
        # effect. We set to zero.
        for dim in range(src_freq.ndim):
            if shape[dim] == 1:
                shifts[dim] = 0

        if return_error:
            return shifts, self._compute_error(CCmax, src_amp, target_amp)
        else:
            return shifts        

    def _shift_stack_inplace(self, stack, shift_list):
        for img_idx in range(stack.shape[2]):
            y_shift = shift_list[0,img_idx]
            x_shift = shift_list[1,img_idx]
            kernel  = op.exp(op.multiply_complex(op._j, op.r2c(2 * np.pi * (self.kx_lin * x_shift + self.ky_lin * y_shift))))
            stack[...,img_idx] = op.convolve_kernel(op.r2c(stack[...,img_idx]), kernel, n_dim=2)[...,0]
        return stack

    def estimate(self, predicted, measured):
        """
        A function to estimate shift error and return the shift correct image stack
        Input parameters:
            - predicted: predicted amplitudes, should be torch array
            - measured: measured amplitudes, should be torch array
        """
        assert predicted.shape == measured.shape
        shift_list = np.zeros((2,measured.shape[2]), dtype="float32")
        err_list = []

        #Change from torch array to numpy array
        flag_predicted_gpu = predicted.is_cuda
        if flag_predicted_gpu:
            predicted = predicted.cpu()

        flag_measured_gpu = measured.is_cuda
        if flag_measured_gpu:
            measured = measured.cpu()        
        
        predicted_np = np.array(predicted.detach())
        measured_np  = np.array(measured.detach())
        
        #For each image, estimate the shift error
        for img_idx in range(measured_np.shape[2]):
            shift, err = self._cross_correlation(predicted_np[...,img_idx], \
                                                 measured_np[...,img_idx], \
                                                 method = self.method, \
                                                 upsample_factor=self.upsample_factor)
            shift_list[:,img_idx] = shift.astype("float32")
            err_list.append(err)
        
        #Change data back to torch tensor format
        if flag_predicted_gpu:
            predicted = predicted.cuda()

        measured_np = torch.tensor(measured_np)
        if flag_measured_gpu:
            measured    = measured.cuda()        
            measured_np = measured_np.cuda()

        #Shift measured image
        measured_np = self._shift_stack_inplace(measured_np, shift_list)
        if any(abs(shift_list > 10.0)):
        	print("Shift too large!", np.max(np.abs(shift_list)))
        	shift_list[:] = 0.0
        return measured_np, torch.tensor(shift_list), torch.tensor(err_list)

class ImageShiftGradientBased(nn.Module):
    """
    A class that solves for shift between measurement and prediction. This uses pytorch autograd, and is gradient based.
    """
    def __init__(self, shape, dtype=torch.float32, device=torch.device('cuda'), **kwargs):
        super(ImageShiftGradientBased, self).__init__()
        pixel_size = 1.0
        self.ky_lin, self.kx_lin = util.generate_grid_2d(shape, pixel_size, flag_fourier=True, dtype=dtype, device=device)

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