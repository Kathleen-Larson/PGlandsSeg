import sys
import numpy as np
from numpy import random as npr
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose


##################################################################################
################################### Anatomical ###################################
##################################################################################

# These functions are intensity based, but focus on specific regions defined
# by the ground truth pineal/pituitary segmentations. The idea is to randomly
# augment the images to represent anatomical variations that may not be
# sufficiently represented in the training dataset (e.g. pineal calcifications or
# the lack of a posterior pituitary bright spot).


class LabelIntensityMatching:
    def __init__(self,
                 label_target:int,     # label to change intensities within (post. pit.)
                 label_reference:int,  # label to sample intensities from (ant. pit.)
                 chance:float=0.5,     # chance of augmentation
    ):
        self.label_target = label_target
        self.label_reference = label_reference
        self.chance = chance if (chance <= 1 and chance >= 0) or chance is not None \
            else Exception("Invalid chance (must be float between 0 and 1)")

        
    def _histogram_cdf(self, x, nbins:int=100, xmin:int=None, xmax:int=None):
        xmin = x.min() if xmin is None else xmin
        xmax = x.max() if xmax is None else xmax
        return self._min_max_norm(torch.cumsum(torch.histc(x, nbins, min=xmin, max=xmax), dim=0))


    def _min_max_norm(self, x, xmin:float=0, xmax:float=1):
        return (xmax - xmin) * (x - x.min()) / (x.max() - x.min()) + xmin


    def _histogram_match(self, ref, targ):
        # Normalize target intensities to reference values
        Hmin = ref.min()
        Hmax = ref.max()
        targ = torch.round(self._min_max_norm(targ, Hmin, Hmax))

        # Histogram matching
        nbins = int(torch.ceil(Hmax - Hmin))
        vals = [x.item() for x in torch.linspace(Hmin, Hmax, nbins+1)][:-1]
        ref_H = self._histogram_cdf(ref, nbins, xmin=Hmin, xmax=Hmax)
        targ_H = self._histogram_cdf(targ, nbins, xmin=Hmin, xmax=Hmax)

        for i, x in enumerate(vals):
            cdf_inds = torch.where(ref_H >= targ_H[i])[0]
            targ[targ==x] = vals[cdf_inds[0]]

        return targ


    def __call__(self, inputs):
        if torch.rand((1)) < self.chance:
            seg = inputs[1].squeeze(1)
            mask_ref_label = torch.where(seg==self.label_reference, True, False)
            mask_targ_label = torch.where(seg==self.label_target, True, False)

            for chn in range(inputs[0].shape[1]):
                img = inputs[0][:, chn, ...]
                img[mask_targ_label] = self._histogram_match(img[mask_ref_label], img[mask_targ_label])
                inputs[0][:, chn, ...] = img

        return inputs



class SimulateCalcification:
    def __init__(self,
                 label=900,                   # label within which to simulate calcification
                 chance:float=0.5,            # chance to perform augmentation
                 calc_mean:list[float]=110.,  # average expected intensity of voxels w/ calcification
                 randomize=True,              # flag to randomize parameters of calcification intensities
    ):
        self.label = label
        self.chance = chance if (chance <= 1 and chance >= 0) or chance is not None \
            else Exception("Invalid chance (must be float between 0 and 1)")
        self.calc_mean = calc_mean if isinstance(calc_mean, list) else [calc_mean]
        self.randomize = randomize
        
        
    def _simulate_calcification(self, x, noise_mean):
        # Create image w/ gaussian noise
        noise_std = random.uniform(0.2, 0.4) * noise_mean if self.randomize else 0.3 * noise_mean
        noise = torch.normal(mean=noise_mean, std=noise_std, size=tuple(x.size()), device=x.device)

        # Replace some voxels in x with calcification
        dr = random.uniform(0.3, 1.0) if self.randomize else 1.0
        dr_mask = torch.rand(x.shape, dtype=x.dtype, device=x.device)

        return torch.where(dr_mask <= dr, noise, x)
        

    def __call__(self, inputs):
        if torch.rand((1)) < self.chance:
            seg = inputs[1].squeeze(1)
            label_mask = torch.where(seg==self.label, True, False)

            for chn in range(inputs[0].shape[1]):
                img = inputs[0][:, chn, ...]
                img[label_mask] = self._simulate_calcification(img[label_mask], self.calc_mean[chn])
                inputs[0][:, chn, ...] = img

        return inputs

    

##################################################################################
################################### Intensity #################################### 
##################################################################################

# These are a set of more general intensity based augmentations that operate on
# the entire volume rather than only within a specific region.

class BiasField:
    def __init__(self,
                 shape_factor:float=0.025,  # ratio of small field to image size
                 max_value:float=1.0,       # max value of bias field
                 std:float=0.3,             # std of bias field
                 randomize:bool=True,       # flag to randomize transform params or use max vals
                 X:int=3                    # number of image dims (e.g. 2=HxW or 3=HxWxD)
    ):
        self.shape_factor = shape_factor
        self.std = std
        self.max_value = max_value
        self.randomize = randomize
        self.X = X

        
    def _apply_bias_field(self, x):
        sz_full = x.shape
        sz_small = torch.ceil(torch.tensor(sz_full) * self.shape_factor).to(torch.int)
        bf_std = random.uniform(0., self.std) if self.randomize else self.std

        bf_small = torch.normal(mean=0., std=bf_std, size=tuple(sz_small), device=x.device)
        bf_full = F.interpolate(bf_small, size=sz_full[-self.X:],
                                mode='trilinear' if self.X == 3 else 'bilinear')
        return x * torch.exp(bf_full)

        
    def __call__(self, inputs):
        inputs[0] = self._apply_bias_field(inputs[0])
        return inputs



class GammaTransform:
    def __init__(self,
                 std:float=0.5,        # std of gamma value for contrast augmentation
                 randomize:bool=True,  # flag to randomize transform params or use max vals
    ):
        self.std = std
        self.randomize = randomize

    def _gamma_transform(self, x):
        gamma = torch.normal(0., self.std, size=[1], device=x.device) if self.randomize \
            else torch.tensor(self.std, device=x.device)
        return x.pow(torch.exp(gamma))
        
    def __call__(self, inputs):
        inputs[0] = self._gamma_transform(inputs[0])
        return inputs
    


class GaussianNoise:
    def __init__(self,
                 std:float=21.0,       # std of gaussian noise (10% of mean WM intensity)
                 randomize:bool=True,  # flag to randomize transform params or use max vals
                 X:int=3               # number of image dims (e.g. 2=HxW or 3=HxWxD)
    ):
        self.std = std
        self.randomize = randomize
        self.X = X

    def _noise(self, x):
        noise_std = random.uniform(0., self.std) if self.randomize else self.std
        noise = torch.normal(mean=0., std=noise_std, size=tuple(x.size()), device=x.device)
        return x + noise
        
    def __call__(self, inputs):
        inputs[0] = self._noise(inputs[0])
        return inputs



    

class MinMaxNorm:
    def __init__(self,
                 minim:float=0.0,                 # min value for normalization
                 maxim:float=1.0,                 # max value for normalization
                 norm_perc:list[float]=[0., 1.],  # lower and upper percentage bounds for robust normalization
                 use_robust:bool=False,
    ):
        self.minim = minim
        self.maxim = maxim
        self.use_robust = use_robust
        self.norm_perc = norm_perc
        
        
    def _min_max_norm(self, x, minim=None, maxim=None):
        m = self.minim if minim is None else minim
        M = self.maxim if maxim is None else maxim

        return (M - m) * (x - x.min()) / (x.max() - x.min()) + m


    def _robust_norm(self, x):
        # Convert percentages to intensities
        shape = x.shape
        flat_shape = shape[0:2] + (torch.tensor([sz for sz in shape[-3:]]).prod(),)
        x_sorted, _ = x.reshape(flat_shape).sort()
        min_val = x_sorted[..., max(int(self.norm_perc[0] * flat_shape[-1]), 0)][0]
        max_val = x_sorted[..., min(int(self.norm_perc[1] * flat_shape[-1]), flat_shape[-1]-1)][0]

        # Robust normalization
        for chn in range(shape[1]):
            x[:,chn,...] = torch.clamp(x[:,chn,...], min=min_val[chn], max=max_val[chn])
        return self._min_max_norm(x)
        
    
    def __call__(self, inputs):
        inputs[0] = self._robust_norm(inputs[0]) if self.use_robust \
             else self._min_max_norm(inputs[0])
        return inputs


    
##################################################################################
#################################### Spatial #####################################
##################################################################################

# These functions perform randomized spatial transforms to deform the images.

class AffineElasticTransform:
    def __init__(self,
                 translation_bounds:[float,list]=0.0,  # max affine translation (fraction of img shape)
                 rotation_bounds:[float,list]=15.0,    # max affine rotation (degrees)
                 shear_bounds:[float,list]=0.012,      # max affine shearing (fraction of img shape)
                 scale_bounds:[float,list]=0.15,       # max affine scaling (fraction of img shape)
                 elastic_shape_factor:float=0.04,      # ratio of small SVF size to full image size
                 elastic_std:float=3.,                 # std of gaussian used to generate SVF
                 n_elastic_steps:int=7,                # number of squaring/integration steps
                 apply_affine:bool=True,               # flag to perform affine transformation
                 apply_elastic:bool=True,              # flag to perform elastic transformation
                 randomize:bool=True,                  # random affine/elastic or use max values?
                 X:int=3,                              # number of image dims (e.g. 2=HxW or 3=HxWxD)
    ):
        self.X = X
        self.apply_affine = apply_affine
        self.apply_elastic = apply_elastic

        # Parse affine transform parameters
        if self.apply_affine:
            def _parse_affine_param(param, full=True, center=0.):
                if isinstance(param, list):
                    if len(param) == self.X: param = [[-p if full else 0., p] for p in param]
                    elif len(param) == self.X * 2: param = param
                else: param = [[-param if full else 0., param]] * X
                shift = torch.ones((self.X, 2), dtype=torch.float) * center
                return torch.tensor(param).reshape(self.X, 2) + shift
            
            self.translation_bounds = _parse_affine_param(translation_bounds)
            self.rotation_bounds = _parse_affine_param(rotation_bounds)
            self.shear_bounds = _parse_affine_param(shear_bounds, center=0.)
            self.scale_bounds = _parse_affine_param(scale_bounds, center=1.)
            
        # Parse elastic transform parameters
        if self.apply_elastic:
            self.elastic_shape_factor = elastic_shape_factor
            self.elastic_std = elastic_std
            self.n_elastic_steps = n_elastic_steps


    def _make_affine(self, x):
        def _sample_params(bounds):
            return torch.rand(self.X) * torch.diff(bounds).squeeze() + bounds[:,0]
        I = torch.eye(self.X+1, device=x.device)

        # Translations
        t = _sample_params(self.translation_bounds).to(x.device)
        T = I.clone()
        T[torch.arange(self.X),-1] = t

        # Shears
        s = torch.cat([_sample_params(self.shear_bounds),
                       _sample_params(self.shear_bounds)], dim=-1).to(x.device)
        Sinds = torch.ones((self.X+1,self.X+1), dtype=torch.bool, device=x.device)
        Sinds[torch.eye(self.X+1, dtype=torch.bool)] = False
        Sinds[-1,:] = False
        Sinds[:,-1] = False
        S = I.clone()
        S[Sinds] = s
        
        # Zooms
        z = _sample_params(self.scale_bounds).to(x.device)
        Z = I.clone()
        Z[torch.arange(self.X), torch.arange(self.X)] = z
        
        # Rotations
        r = _sample_params(self.rotation_bounds).to(x.device) * torch.pi/180
        c, s = [torch.cos(r), torch.sin(r)]
        if self.X == 2:
            r = [0]
            R = I.clone()
            R[torch.tensor([0,1,0,1]),torch.tensor([0,0,1,1])] = torch.tensor([c, s, -s, c])
        else:            
            [R1, R2, R3] = [I.clone(), I.clone(), I.clone()]
            R1[torch.tensor([1,2,1,2]),torch.tensor([1,1,2,2])] = torch.tensor([c[0], s[0], -s[0], c[0]]).to(x.device)
            R2[torch.tensor([0,2,0,2]),torch.tensor([0,0,2,2])] = torch.tensor([c[1], s[1], -s[1], c[1]]).to(x.device)
            R3[torch.tensor([0,1,0,1]),torch.tensor([0,0,1,1])] = torch.tensor([c[2], s[2], -s[2], c[2]]).to(x.device)
            R = R3 @ R2 @ R1

        ## Convert affine matrix to displacement field
        aff = T @ R @ Z @ S

        grid = torch.stack(torch.meshgrid([torch.arange(-(s-1)/2, s/2, dtype=x.dtype, device=x.device) \
                                           for s in x.shape[-self.X:]], indexing='ij'), dim=-1)
        coords = torch.cat([grid.view(-1, self.X), torch.ones((grid.numel()//self.X,1),
                                                              device=x.device)], dim=-1)
        aff_coords = (coords @ aff.transpose(0,1)).view(*x.shape[-self.X:], 4)
        disp = 2 * aff_coords[...,:self.X] / (torch.tensor(x.shape[-self.X:]) - 1).to(x.device)

        return disp

    
    def _make_elastic(self, x):
        # Get field shapes
        sz_full = x.size()
        sz_small = torch.cat([torch.ceil(torch.tensor(sz_full) * self.elastic_shape_factor).to(torch.int),
                              torch.tensor([self.X])])
        sz_half = torch.cat([torch.ceil(torch.tensor(sz_full) * 0.5).to(torch.int),
                             torch.tensor([self.X])])

        # Create SVF
        svf_small = torch.normal(mean=0., std=torch.rand(1).item(), size=tuple(sz_small), device=x.device)
        svf_half =  torch.stack([F.interpolate(svf_small[...,i], size=tuple(sz_half[-self.X-1:-1]),
                                               mode='trilinear' if self.X == 3 else 'bilinear') \
                                 for i in range(self.X)], dim=-1)
        
        # Scaling and squaring
        svf_half = (svf_half / (2 ** self.n_elastic_steps)).squeeze(1).movedim(-1,1)
        grid_half = torch.stack(torch.meshgrid([torch.arange(s, dtype=svf_half.dtype, device=x.device) \
                                                for s in svf_half.shape[-self.X:]], indexing='ij'),
                                dim=-1).movedim(-1,0).unsqueeze(0)
        weights = 2/torch.tensor(grid_half.shape[-self.X:], device=x.device)

        for _ in range(self.n_elastic_steps - 1):
            grid_interp = (svf_half + grid_half).movedim(1,-1) * weights
            svf_half += F.grid_sample(svf_half, grid_interp, align_corners=True)
        
        # Interpolate to full size
        elastic = F.interpolate(svf_half, size=sz_full[-self.X:], align_corners=True, mode='trilinear')
       
        return elastic.movedim(1,-1)


    def __call__(self, inputs):
        if self.apply_affine and self.apply_elastic:
            A = self._make_affine(inputs[0]).unsqueeze(0)
            E = self._make_elastic(inputs[0])
            T = A + E
        elif self.apply_affine:
            T = self._make_affine(inputs[0]).unsqueeze(0)
        elif self.apply_elastic:
            T = self._make_elastic(inputs[0])
        else:
            T = None

        inputs = inputs if T is None else \
            [F.grid_sample(x, T.permute(0, 3, 2, 1, 4), align_corners=True, mode='bilinear') for x in inputs]

        return inputs

    

class CropPatch:
    def __init__(self,
                 patch_size:[int, list]=None,   # size of crop patch
                 randomize:bool=True,           # randomize crop patch or centered in bbox?
                 use_label_template:bool=True,  # use a template to determine bbox for cropping?
                 X:int=3,                       # number of image dims (e.g. 2=HxW or 3=HxWxD)
    ):
        self.X = X
        self.patch_size = [patch_size] * X if isinstance(patch_size, int) else patch_size
        self.use_label_template = use_label_template

        if patch_size is not None:
            if randomize:
                self._get_bounds = self._get_random_crop_bounds
            else:
                self._get_bounds = self._get_center_crop_bounds

                
    def _get_bounding_box(self, vol, bffr:int=4):
        vol = torch.squeeze(vol)
        bbox = [[0, vol.shape[i]-1] for i in range(len(vol.shape))]

        while vol[bbox[0][0],:,:].sum() == 0: bbox[0][0] += 1
        while vol[bbox[0][1],:,:].sum() == 0: bbox[0][1] -= 1
        while vol[:,bbox[1][0],:].sum() == 0: bbox[1][0] += 1
        while vol[:,bbox[1][1],:].sum() == 0: bbox[1][1] -= 1
        while vol[:,:,bbox[2][0]].sum() == 0: bbox[2][0] += 1
        while vol[:,:,bbox[2][1]].sum() == 0: bbox[2][1] -= 1

        bbox = [[bb[0] - (bffr), bb[1] + (bffr)] for bb in bbox]
        return bbox


    def _get_center_crop_bounds(self, vol_sz, bbox:list=None):
        vol_sz = vol_sz[-self.X:]
        patch_sz = self.patch_size

        if bbox is not None:
            center = [(bb[0] + bb[1])//2 for bb in bbox]
        else:
            center = [vs//2 for vs in vol_sz]
        bounds = [[c - ps//2, c + ps//2] for c, ps in zip(center, patch_sz)]

        for i in range(self.X):
            if bounds[i][0] < 0:  shift = bounds[i][0]
            elif bounds[i][1] > vol_sz[i]:  shift = bounds[i][1] - vol_sz[i]
            else:  shift = 0
            bounds[i] = [bounds[i][0] - shift, bounds[i][1] - shift]

        return bounds


    def _get_random_crop_bounds(self, vol_sz, bbox:list=None, return_bounds=False):
        vol_sz = vol_sz[-self.X:]
        patch_sz = self.patch_size

        if bbox is not None:
            rand_bounds = [[max([0, bbox[i][1] - patch_sz[i] + 1]),
                           min([vol_sz[i] - patch_sz[i], bbox[i][0]])] for i in range(self.X)]
        else:
            rand_bounds = [[0, vol_sz[i] - patch_sz[i]] for i in range(self.X)]

        start_idx = [rand_bounds[i][1] if rand_bounds[i][1] <= rand_bounds[i][0] \
                     else npr.randint(rand_bounds[i][0], rand_bounds[i][1]) for i in range(self.X)]
        bounds = [(start_idx[i], start_idx[i] + patch_sz[i]) for i in range(self.X)]

        return bounds
    

    def _apply_crop(self, vol, bounds):
        if self.X == 2:
            h, w = bounds
            crop = vol[..., h[0]:h[1], w[0]:w[1]]
        elif self.X >= 3:
            h, w, d = bounds
            crop = vol[..., h[0]:h[1], w[0]:w[1], d[0]:d[1]]
        else:
            print(f'Invalid X (X=={self.X}')

        return crop

    
    def __call__(self, inputs, return_bounds=False):
        if self.patch_size is not None:
            bbox = None if not self.use_label_template else self._get_bounding_box(inputs[-1], bffr=2)
            bounds = self._get_bounds(inputs[0].shape, bbox)
            inputs = [self._apply_crop(x, bounds) for x in inputs]
        
        return inputs



class FlipTransform:
    def __init__(self,
                 flip_axis:int=None, # axis to flip
                 chance:float=0.5,   # probability of flipping
                 X=3                 # number of image dims
    ):
        self.X = X
        self.chance = chance if (chance <= 1 and chance >= 0) or chance is not None \
            else Exception("Invalid chance (must be float between 0 and 1)")
        self.flip_axis = flip_axis if isinstance(flip_axis, int) or flip_axis is None \
            else Exception("Invalid flip_axis (must be int < number of image dims)")

    def __call__(self, inputs):
        if torch.rand((1)) < self.chance:
            inputs = [torch.fliplr(x.transpose(1,self.flip_axis)).transpose(1,self.flip_axis) \
                      for x in inputs]
        return inputs



##################################################################################
###################################### Misc ######################################
##################################################################################


class AssignOneHotLabels:
    def __init__(self,
                 label_values:list=None, # Label values to encode
                 X:int=3,                # Number of image dims (e.g. 2=HxW or 3=HxWxD)
    ):
        self.label_values = label_values
        self.X = X


    def _one_hot_encode(self, x):
        if self.label_values == None:
            self.label_values = torch.unique(x)

        onehot = torch.zeros(x.shape).to(x.device)
        if self.X == 4:
            onehot = onehot.repeat(1,len(self.label_values),1,1,1,1)
        elif self.X == 3:
            onehot = onehot.repeat(1,len(self.label_values),1,1,1)
        elif self.X == 2:
            onehot = onehot.repeat(1,len(self.label_values),1,1)

        x = x.squeeze()
        for i in range(0, len(self.label_values)):
            onehot[:,i,...] = x==self.label_values[i]

        return onehot

    
    def __call__(self, inputs):
        seg_ind = 1
        inputs[seg_ind] = self._one_hot_encode(inputs[seg_ind])
        return inputs



class ComposeTransforms:
    def __init__(self, transform_list):
        self.transforms = [t for t in transform_list if t is not None]
        
    def __call__(self, inputs):
        for T in self.transforms:
            if T is not None:
                inputs = T(inputs)
        return inputs
