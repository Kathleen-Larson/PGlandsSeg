import sys
import numpy as np
from numpy import random as npr
import math
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

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
        self.check_patch_tol = 0.1
        
        if randomize:
            npr.seed()
            self.get_bounds = self._get_random_bounds
            self.check_patch = True
        else:
            self.get_bounds = self._get_center_bounds
            self.check_patch = False
            

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

        
    
    def _get_patch(self, vol, bounds):
        if self.X == 2:
            h, w = bounds
            crop = vol[..., h[0]:h[1], w[0]:w[1]]
        elif self.X >= 3:
            h, w, d = bounds
            crop = vol[..., h[0]:h[1], w[0]:w[1], d[0]:d[1]]
        else:
            print(f'Invalid X (X=={self.X}')

        return crop
                
        
    def __call__(self, img, seg):
        crop_seg = torch.zeros(list(seg.shape[:2]) + list(self.patch_size))
        
        if self.check_patch:
            while abs(crop_seg[:, 1:, ...].sum().item() - seg[:, 1:, ...].sum().item()) > self.check_patch_tol:
                bounds = self.get_bounds(seg.shape)
                crop_seg = self._get_patch(seg, bounds)
            img = self._get_patch(img, bounds)
            seg = self._get_patch(seg, bounds)
        else:
            bounds = self.get_bounds(seg.shape)
            img = self._get_patch(img, bounds)
            seg = self._get_patch(seg, bounds)
     
        return img, seg



class RandomElasticAffineCrop:
    def __init__(self,
                 translation_bounds:[float,list]=0.0,
                 rotation_bounds:[float,list]=15,
                 shear_bounds:[float,list]=0.012,
                 scale_bounds:[float,list]=0.15,
                 max_elastic_displacement:[float,list]=0.15,
                 n_elastic_control_pts:int=5,
                 n_elastic_steps:int=0,
                 order:int=3,
                 X:int=3,
                 **kwargs
    ):
        self.X = X
        if isinstance(translation_bounds, list): assert len(translation_bounds) == X
        if isinstance(rotation_bounds, list): assert len(rotation_bounds) == X
        if isinstance(shear_bounds, list): assert len(shear_bounds) == X
        if isinstance(scale_bounds, list): assert len(scale_bounds) == X

        self.translation_bounds = [translation_bounds] * X \
            if isinstance(translation_bounds, float) else translation_bounds
        self.rotation_bounds = [rotation_bounds] * X \
            if isinstance(rotation_bounds, float) else rotation_bounds
        self.shear_bounds = [shear_bounds]  * X \
            if isinstance(shear_bounds, float) else shear_bounds
        self.scale_bounds = [scale_bounds]  * X \
            if isinstance(scale_bounds, float) else scale_bounds
        self.max_elastic_displacement = [max_elastic_displacement] * X \
            if isinstance(max_elastic_displacement, float) else max_elastic_displacement
        self.n_elastic_control_pts = n_elastic_control_pts
        self.n_elastic_steps = n_elastic_steps
        
        self.transform = cc.RandomAffineElasticTransform(translations=self.translation_bounds,
                                                         rotations=self.rotation_bounds,
                                                         shears=self.shear_bounds,
                                                         zooms=self.scale_bounds,
                                                         dmax=self.max_elastic_displacement,
                                                         shape=self.n_elastic_control_pts,
                                                         steps=self.n_elastic_steps,
        )
        
        
    def __call__(self, img, seg):
        if len(img.shape) > self.X + 1:
            for b in range(img.shape[0]):
                img[b,:], seg[b,:] = self.transform(img[b,:], seg[b,:])
        else:
            img, seg = self.transform(img, seg)
        return img, seg



class RandomLRFlip:
    def __init__(self, axis:int, chance:float=0.5):
        self.chance = chance if chance >= 0 and chance <= 1 \
            else Exception("Invalid chance (must be float between 0 and 1)")
        self.transform = cc.FlipTransform(axis=axis)
        

    def __call__(self, img, seg):
        img, seg = cc.MaybeTransform(self.transform, self.chance)(img, seg)
        return img, seg
        
    

class MinMaxNorm:
    def __init__(self, minim:float=0, maxim:float=1):
        self.minim = minim
        self.maxim = maxim

    def __call__(self, img, seg):
        i_min = torch.min(img)
        i_max = torch.max(img)
        o_min = self.minim
        o_max = self.maxim

        img = (o_max - o_min) * (img - i_min) / (i_max - i_min) + o_min

        return img, seg

               

class ContrastAugmentation:
    """
    def __init__(self, gamma_range:list=(0.5, 2)):
        self.gamma_range = gamma_range if len(gamma_range)==2 \
            else Exception("Invalid gamma_range (must be (min max))")
        #self.transform = cc.RandomGammaTransform(gamma=gamma_range)

    def __call__(self, img, seg):
        breakpoint()
        #img = self.transform(img)
        img = _gamma_transform(img)
        return img, seg
    """
    def __init__(self, gamma_std:float=0.5):
        self.stdev = gamma_std
        npr.seed()
        
    def _gamma_transform(self, img):
        gamma = npr.normal(0., self.stdev)
        img = img.pow(np.exp(gamma))
        return img
        
    def __call__(self, img, seg):
        img = self._gamma_transform(img)
        return img, seg
    

    
class BiasField:
    def __init__(self, shape:int=8, v_max:list=1, order:int=3):
        self.shape = shape if isinstance(shape, int)\
            else Exception("Invalid shape (must be int)")
        self.v_max = v_max if isinstance(v_max, int)\
            else Exception("Invalid v_max (must be int)")
        self.order = order if isinstance(order, int)\
            else Exception("Invalid order (must be int)")
        
        self.transform = cc.RandomMulFieldTransform(shape=shape,
                                                    vmax=v_max,
                                                    order=order,
                                                    shared=False)

    def __call__(self, img, seg):
        img = self.transform(img)
        return img, seg

    

class GaussianNoise:
    def __init__(self, sigma:float=0.1):
        self.sigma = sigma
        self.transform = cc.RandomGaussianNoiseTransform(sigma=sigma)

    def __call__(self, img, seg):
        img = self.transform(img)
        return img, seg



class AssignOneHotLabels():
    def __init__(self, label_values:list=None, X:int=3, index=0):
        self.label_values = label_values
        self.X = X
        self.index = index

        
    def __call__(self, img, seg):
        if self.label_values == None:
            self.label_values = torch.unique((seg))
        
        onehot = torch.zeros(seg.shape).to(seg.device)
        if self.X == 4:
            onehot = onehot.repeat(1,len(self.label_values),1,1,1,1)
        elif self.X == 3:
            onehot = onehot.repeat(1,len(self.label_values),1,1,1)
        elif self.X == 2:
            onehot = onehot.repeat(1,len(self.label_values),1,1)

        seg = torch.squeeze(seg)
        for i in range(0, len(self.label_values)):
            onehot[:,i,...] = seg==self.label_values[i]

        return img, onehot.type(torch.float32)
