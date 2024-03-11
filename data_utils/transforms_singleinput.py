import sys
import numpy as np
from numpy import random as npr
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import cornucopia as cc #https://github.com/balbasty/cornucopia/tree/feat-psf-slicewise


class Compose(transforms.Compose):
    def __init__(self, transforms, gpuindex=1):
        super().__init__(transforms)
        self.gpuindex = gpuindex

    def __call__(self, *args, cpu=True, gpu=True, **kwargs):
        if cpu:
            for t in self.transforms[:self.gpuindex]:
                args = t(*args) if t is not None else args
        if gpu:
            for t in self.transforms[self.gpuindex:]:
                args = t(*args) if t is not None else args
                
        return args



class GetPatch:
    def __init__(self, patch_size:[int, list], X:int, randomize:bool=False):
        self.X = X
        self.patch_size = [patch_size] * X if isinstance(patch_size, int) else patch_size
        self.randomize = randomize

        if self.randomize:  npr.seed()


    def _get_center_bounds(self, in_sz):
        vol_sz = list(in_sz[-self.X:])
        patch_sz = self.patch_size
        bounds = [((vol_sz[i] - patch_sz[i])//2, vol_sz[i] - (vol_sz[i] - patch_sz[i])//2) for i in range(self.X)]
        return bounds


    def _get_random_bounds(self, in_sz):
        vol_sz = list(in_sz[-self.X:])
        patch_sz = self.patch_size
        idx = [npr.randint(0, (vol_sz[i] - patch_sz[i])) for i in range(self.X)]
        bounds = [(idx[i], idx[i] + patch_sz[i]) for i in range(self.X)]
        return bounds
        
    
    def _get_patch(self, vol):
        get_bounds = self._get_random_bounds if self.randomize else self._get_center_bounds
        if self.X == 2:
            h, w = get_bounds(vol.shape)
            crop = vol[..., h[0]:h[1], w[0]:w[1]]
        elif self.X >= 3:
            h, w, d = get_bounds(vol.shape)
            crop = vol[..., h[0]:h[1], w[0]:w[1], d[0]:d[1]]
        else:
            print(f'Invalid X (X=={self.X}')
        return crop
                
        
    def __call__(self, img):
        #print('GetPatch in_shape', img.shape)
        img = self._get_patch(img)
        #print('GetPatch out_shape', img.shape)
        return img



class MinMaxNorm:
    def __init__(self, minim:float=0, maxim:float=1):
        self.minim = minim
        self.maxim = maxim

    def __call__(self, img):
        i_min = torch.min(img)
        i_max = torch.max(img)
        o_min = self.minim
        o_max = self.maxim
        #print('MinMaxNorm in_shape', img.shape)
        img = (o_max - o_min) * (img - i_min) / (i_max - i_min) + o_min
        #print('MinMaxNorm out_shape', img.shape)
        return img



class ResampleImage:
    def __init__(self, input_shape, input_res,
                 output_shape, output_res
    ):
        self.resample_shape = ((input_res * input_shape) / output_res).astype(int)
        self.pad_size = np.zeros(len(self.resample_shape)*2, dtype=int)
        for i in range(len(self.resample_shape)):
            if self.resample_shape[i] % 2!= 0:
                self.pad_size[2*i] = np.floor((output_shape[i] - self.resample_shape[i])/2)
                self.pad_size[2*i+1] = np.ceil((output_shape[i] - self.resample_shape[i])/2)
            else:
                self.pad_size[(2*i):(2*i+1)+1] = (output_shape[i] - self.resample_shape[i])/2

        
    def __call__(self, img):
        #print('ResampleImage in_shape', img.shape)
        img = F.interpolate(img, size=tuple(self.resample_shape), mode='trilinear')
        img = F.pad(img, pad=tuple(self.pad_size[::-1]))
        #print('ResampleImage out_shape', img.shape)
        return img
    
                                             
