"""
Transform functions for Tomography in Numpy, Scipy, Torch, and Skimage
Estimates affine transform between measured image and predicted image
https://github.com/scikit-image/scikit-image

David Ren      david.ren@berkeley.edu

Dec 28, 2020
"""

import numpy as np
import torch
from skimage.registration import optical_flow_tvl1
from skimage import transform
import scipy.optimize as sop
from pystackreg import StackReg

class ImageTransformOpticalFlow():
    """
    Class written to register stack of images for AET.
    Uses correlation based method to determine subpixel shift between predicted and measured images.
    Input parameters:
        - shape: shape of the image
    """ 
    def __init__(self, shape, method="optical_flow"):
        self.shape = shape
        self.x_lin, self.y_lin = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        self.xy_lin = np.concatenate((self.x_lin[np.newaxis,], self.y_lin[np.newaxis,])).astype('float32')
        self.sr = StackReg(StackReg.RIGID_BODY)

    def _estimate_single(self, predicted, measured):
        assert predicted.shape == self.shape
        assert measured.shape == self.shape
        aff_mat = self.sr.register(measured, predicted)
        tform = transform.AffineTransform(matrix = aff_mat)
        measured_warp = transform.warp(measured, tform.inverse, cval = 1.0, order = 5)
        transform_final = aff_mat.flatten()[0:6]
        return measured_warp, transform_final

    def estimate(self, predicted_stack, measured_stack):
        assert predicted_stack.shape == measured_stack.shape
        transform_vec_list = np.zeros((6,measured_stack.shape[2]), dtype="float32")

        #Change from torch array to numpy array
        flag_predicted_gpu = predicted_stack.is_cuda
        if flag_predicted_gpu:
            predicted_stack = predicted_stack.cpu()

        flag_measured_gpu = measured_stack.is_cuda
        if flag_measured_gpu:
            measured_stack = measured_stack.cpu()        
        
        predicted_np = np.array(predicted_stack.detach())
        measured_np  = np.array(measured_stack.detach())
        
        #For each image, estimate the affine transform error
        for img_idx in range(measured_np.shape[2]):
            measured_np[...,img_idx], transform_vec = self._estimate_single(predicted_np[...,img_idx], \
                                                                      measured_np[...,img_idx])
            transform_vec_list[...,img_idx] = transform_vec
        
        #Change data back to torch tensor format
        if flag_predicted_gpu:
            predicted_stack = predicted_stack.cuda()

        measured_np = torch.tensor(measured_np)
        if flag_measured_gpu:
            measured_stack  = measured_stack.cuda()        
            measured_np     = measured_np.cuda()

        return measured_np, torch.tensor(transform_vec_list)
