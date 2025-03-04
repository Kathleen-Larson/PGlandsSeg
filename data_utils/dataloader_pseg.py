import os
from glob import glob
from pathlib import Path
import random

import logging
import pandas as pd
import nibabel as nib
import numpy as np
import surfa as sf
import torch

from scipy import ndimage

from torch.utils.data import Dataset
from torchvision import transforms
import freesurfer as fs

from models import augmentations as aug


##############

def _get_crop_window(vol, patch_sz=[96, 96, 96], bffr:int=2):
    vol_sz = vol.shape
    bbox = [[0, vol_sz[i]-1] for i in range(len(vol_sz))]

    while np.sum(vol[bbox[0][0],:,:]) == 0: bbox[0][0] += 1
    while np.sum(vol[bbox[0][1],:,:]) == 0: bbox[0][1] -= 1
    while np.sum(vol[:,bbox[1][0],:]) == 0: bbox[1][0] += 1
    while np.sum(vol[:,bbox[1][1],:]) == 0: bbox[1][1] -= 1
    while np.sum(vol[:,:,bbox[2][0]]) == 0: bbox[2][0] += 1
    while np.sum(vol[:,:,bbox[2][1]]) == 0: bbox[2][1] -= 1

    bbox = [[bb[0] - (bffr), bb[1] + (bffr)] for bb in bbox]
    center = [(bb[0] + bb[1])//2 for bb in bbox]

    bounds = [[c - ps//2, c + ps//2] for c, ps in zip(center, patch_sz)]
    for i in range(len(bbox)):
        if bounds[i][0] < 0:  shift = bounds[i][0]
        elif bounds[i][1] > vol_sz[i]:  shift = bounds[i][1] - vol_sz[i]
        else:  shift = 0
        bounds[i] = [bounds[i][0] - shift, bounds[i][1] - shift]

    return bounds
    


##############

class PituitaryPinealDataset(Dataset):
    def __init__(self,
                 data_inds:list,
                 n_input:int=1,
                 n_class:int=1,
                 data_file=None,
                 data_labels:list=None,
                 base_augmentations=None,
                 full_augmentations=None,
                 has_ground_truth:bool=True,
                 lut_path:str=None,
                 use_template_labels:bool=False,
                 device=None,
    ):
        # Initialize
        self.n_input = n_input
        self.n_class = n_class
        self.device = 'cpu' if device is None else device

        self.has_ground_truth = has_ground_truth
        self.use_template_labels = use_template_labels
        
        self.data_inds = data_inds        
        self.data_labels = data_labels
        self.lut = sf.load_label_lookup(lut_path) if lut_path is not None else None
        
        # Set up transforms
        self.base_augmentations = aug.ComposeTransforms(base_augmentations)
        self.full_augmentations = aug.ComposeTransforms(full_augmentations)
        
        # Load data
        if not os.path.isfile(data_file):
            raise ValueError(f'File {data_file} does not exist')
        if data_file is None:
            raise Exception('Must input data_file')

        df = pd.read_csv(data_file, header=None)
        self.image_files = df.iloc[data_inds,0:n_input].values.tolist()
        self.label_files = df.iloc[data_inds,n_input].values.tolist() if has_ground_truth else None
        self.template_files = None if not use_template_labels \
            else df.iloc[data_inds,-1].values.tolist()

        
    def __len__(self):
        return len(self.image_files)

    
    def __numinput__(self) -> int:
        return self.n_input
    
    
    def __numclass__(self) -> int:
        return self.n_class
    
    
    def _load_volume(self, path, dtype, shape=(256,256,256), res=1.0, orientation='RAS',
                     is_labels=False, return_orig=False):
        img_raw = sf.load_volume(str(path))
        img = img_raw.conform(shape=shape,
                              voxsize=res,
                              orientation=orientation,
                              dtype=dtype,
                              method='nearest' if is_labels else 'linear'
        )        
        return img if not return_orig else [img_raw, img]
    

    def _convert_labels(self, img, labels):
        img_convert = img.copy()
        for i in range(len(labels)):
            img_convert[img==i] = labels[i]

        return img_convert


    def _largest_connected_components(self, y, vals, bgval=0):
        vals = [i for i in vals if i != bgval]
        y_cc = np.tile(np.zeros(y.shape), (len(vals)+1,1,1,1))

        for j in range(len(vals)):
            y_j = np.squeeze(np.where(y==vals[j], 1, 0))
            y_j_cc, n_cc = ndimage.label(y_j, np.ones((3,3,3)))

            if n_cc > 1:
                cc_vals = np.unique(y_j_cc)[1:]
                cc_counts = np.array([(y_j_cc==i).sum() for i in cc_vals])
                try:
                    largest_cc_val = cc_vals[cc_counts==cc_counts.max()].item()
                except:
                    largest_cc_val = cc_vals[np.array(cc_counts==cc_counts.max(), dtype=int)[0]].item()
            else:
                largest_cc_val = 1
            y_cc[j+1, ...] = np.where(y_j_cc==largest_cc_val, vals[j], 0)

        y_cc = np.sum(y_cc, axis=0, dtype=y.dtype)
        if len(np.unique(y_cc)) != len(np.unique(y)): breakpoint()
        return y_cc
    


    def _pad_volume(self, vol, ref_path:str=None, crop_type='bbox'):
        ref = self._load_volume(ref_path, dtype=int)
        if crop_type == 'bbox':
            crop_window = _get_crop_window(ref.data, bffr=2)
            pad_width = [[w[0], rs - w[1]] for w, rs in zip(crop_window, ref.shape)]
        elif crop_type == 'center':
            pad_width = [[(rs - vs)//2, (rs - vs)//2] for vs, rs in zip(vol.shape, ref.shape)]

        return np.pad(vol, pad_width=pad_width)


    def _rescale_volume(self, vol, ref):
        mask = np.where(vol > 0, 1, 0)
        ref_masked = mask * ref
        vol = (ref_masked.max() - ref_masked.min()) * (vol - vol.min()) / (vol.max() - vol.min())
        return vol + ref_masked.min()


    def _conform_to_original_space(self, img, idx, geom_idx, dtype=np.float32, is_labels=False):
        ref_path = self.template_files[idx] if is_labels else self.image_files[idx][geom_idx]
        ref, ref_conform = self._load_volume(ref_path, dtype=int if is_labels else dtype, return_orig=True)

        img = self._pad_volume(img, ref_path=self.template_files[idx] if self.use_template_labels \
                               else self.image_files[idx][geom_idx],
                               crop_type='bbox' if self.use_template_labels else 'center'
        )
        img = self._rescale_volume(img, ref_conform) if not is_labels else img
        img = sf.Volume(img, geometry=ref_conform.geom)
        
        img = img.conform(shape=ref.geom.shape,
                          voxsize=ref.geom.voxsize,
                          orientation=ref.geom.orientation,
                          method='nearest' if is_labels else 'linear'
        )
        
        return img


    def _save_output(self, img, path, idx, geom_idx, dtype, crop_type:str='bbox', conform2orig=True,
                     is_labels=False, is_onehot=False, convert_labels=False):
        output = torch.argmax(img, dim=1).squeeze().cpu().numpy().astype(dtype) \
            if is_onehot else img.squeeze().cpu().numpy().astype(dtype)
        output = self._convert_labels(output, self.data_labels) \
            if convert_labels else output
        output = self._largest_connected_components(output, self.data_labels) \
            if is_labels else output
        output = self._conform_to_original_space(img=output, idx=idx, geom_idx=geom_idx,
                                                 dtype=dtype, is_labels=is_labels) \
                                                 if conform2orig else sf.Volume(output)
        if self.lut is not None:  output.labels = self.lut        
        output.save(path)
        
    
    def __getitem__(self, idx, augment:bool=False):
        image_paths = self.image_files[idx]
        images = [self._load_volume(Path(path), dtype=np.float32) for path in image_paths]
        images = torch.stack([torch.tensor(image.data) for image in images], dim=0)

        data = [images]
        
        if self.has_ground_truth:
            label_path = self.label_files[idx]
            label = self._load_volume(Path(label_path), dtype=np.float32, is_labels=True)
            label = torch.tensor(label.data).unsqueeze(0)

            data += [label]
            
        if self.use_template_labels:
            template_path = self.template_files[idx]
            template = self._load_volume(Path(template_path), dtype=np.float32, is_labels=True)
            template = torch.tensor(template.data).unsqueeze(0)

            data += [template]
            
        return data, idx



def augmentation_setup(aug_config:str=None,
                       data_types:list[str]=None,
                       label_values:list[int]=None,
                       crop_patch_size:list[int]=None,
                       use_crop_template:bool=False,
                       prob_lr_flip:int=0.5,
                       prob_augment_label_intensities:float=0.5,
                       lut_path:str=None,
                       random_crop:bool=False,
                       random_aug:bool=True,
                       apply_postpit_aug:bool=False,
                       apply_pineal_aug:bool=False,
                       apply_robust_normalization:bool=True,
                       has_ground_truth:bool=True,
):
    X = 3
    
    ## Base transforms
    onehot = aug.AssignOneHotLabels(label_values=label_values, X=X) if has_ground_truth \
        else None
    
    center_crop = aug.CropPatch(patch_size=crop_patch_size,
                                randomize=False,
                                use_label_template=use_crop_template,
                                X=X
    )
    norm = aug.MinMaxNorm(minim=0.,
                          maxim=1.,
                          norm_perc=[0., 0.99],
                          use_robust=True if apply_robust_normalization else False
    )
    base_augmentations = [onehot, center_crop, norm]
    

    ## Full augmentations
    if prob_augment_label_intensities > 0.0 and lut_path is not None:
        lut_df = pd.read_csv(lut_path, sep='\s+', index_col=1, header=None)

        # Anterior/posterior pituitary histogram matching
        if apply_postpit_aug:
            ref_label = lut_df.loc['Pituitary-Ant'][0]
            targ_label = lut_df.loc['Pituitary-Post'][0]
            postpit_aug = aug.LabelIntensityMatching(label_target=targ_label,
                                                            label_reference=ref_label,
                                                            chance=prob_augment_label_intensities,
            )
        else:
            postpit_aug = None

        # Pineal calcification
        if apply_pineal_aug:
            all_data_types = ['t1', 't2', 'pd', 'qt1']
            vals = [45, 100, 70, 150]
            calc_means = [None] * len(data_types)
            for i, x in enumerate(all_data_types):
                if x in data_types:
                    j = data_types.index(x)
                    calc_means[j] = vals[i]

            label = lut_df.loc['Pineal'][0]
            pineal_aug = aug.SimulateCalcification(label=label,
                                                   chance=prob_augment_label_intensities,
                                                   calc_mean=calc_means,
                                                   randomize=True
            )
        else:
            pineal_aug = None
    else:
        postpit_aug = None
        pineal_aug = None

    
    if aug_config is None:
        full_augmentations = [onehot, center_crop, norm]

    else:
        df = pd.read_table(aug_config, delimiter='=', header=None)
        
        # Left-right flipping
        flip = aug.FlipTransform(flip_axis=X-1, chance=prob_lr_flip)

        # Spatial deformation
        translation_bounds = float(df.loc[df.iloc[:,0]=="translation_bounds",1].item())
        rotation_bounds = float(df.loc[df.iloc[:,0]=="rotation_bounds",1].item())
        shear_bounds = float(df.loc[df.iloc[:,0]=="shear_bounds",1].item())
        scale_bounds = float(df.loc[df.iloc[:,0]=="scale_bounds",1].item())
        
        elastic_shape_factor = float(df.loc[df.iloc[:,0]=="elastic_shape_factor",1].item())
        elastic_std = float(df.loc[df.iloc[:,0]=="elastic_std",1].item())
        n_elastic_steps = int(df.loc[df.iloc[:,0]=="n_elastic_integration_steps",1].item())
        
        spatial = aug.AffineElasticTransform(translation_bounds=translation_bounds,
                                             rotation_bounds=rotation_bounds,
                                             shear_bounds=shear_bounds,
                                             scale_bounds=scale_bounds,
                                             elastic_shape_factor=elastic_shape_factor,
                                             elastic_std=elastic_std,
                                             n_elastic_steps=n_elastic_steps,
                                             apply_affine=True,
                                             apply_elastic=True,
                                             randomize=True if random_aug else False,
                                             X=X,
        )

        # Random patch
        rand_crop = aug.CropPatch(patch_size=crop_patch_size,
                                  randomize=True if random_aug else False,
                                  use_label_template=use_crop_template,
                                  X=X
        )
        
        # Bias field
        bias_shape_factor = float(df.loc[df.iloc[:,0]=="bias_shape_factor",1].item())
        bias_std = float(df.loc[df.iloc[:,0]=="bias_std",1].item())
        bias_max = float(df.loc[df.iloc[:,0]=="bias_max",1].item())
        
        bias = aug.BiasField(shape_factor=bias_shape_factor,
                             max_value=bias_max,
                             std=bias_std,
                             randomize=True if random_aug else False,
                             X=X
        )

        # Gaussian noise
        noise_std = float(df.loc[df.iloc[:,0]=="noise_std",1].item())
        noise = aug.GaussianNoise(std=noise_std,
                                  randomize=True if random_aug else False
        )

        # Gamma transform
        gamma_std = float(df.loc[df.iloc[:,0]=="gamma_std",1].item())
        gamma = aug.GammaTransform(std=gamma_std,
                                   randomize=True if random_aug else False
        )
        
        full_augmentations = [postpit_aug, pineal_aug,           # anatomical
                              flip, onehot, spatial, rand_crop,  # spatial
                              bias, noise, norm, gamma]          # intensity
        
        
    return base_augmentations, full_augmentations

        

def get_inds(input_data_files:str,
             data_inds_config:str=None,
             n_data_splits:int=1,
             n_include:int=None,
             randomize:bool=False,
             has_ground_truth:bool=True,
             has_template_labels:bool=True,
):
    data_df = pd.read_csv(input_data_files, header=None)
    uniqueIDs = data_df.iloc[:,0].unique()
    n_unique = uniqueIDs.size
    
    [n_subjects, n_inputs] = data_df.shape
    n_inputs = n_inputs - 1 if has_ground_truth else n_inputs
    n_inputs = n_inputs - 1 if has_template_labels else n_inputs

    if n_include is not	None:
        if n_include < n_unique:
            n_unique = n_include
            uniqueIDs = uniqueIDs[0:n_unique]
        else:
            print(f'n_include ({n_include}) >= n_subjects ({n_subjects}) --> using entire dataset')

    if randomize:  random.shuffle(uniqueIDs)

    inds_paired = [None] * n_unique
    for i in range(n_unique):
        inds_paired[i] = data_df.loc[data_df.iloc[:,0]==uniqueIDs[i]].index.tolist()

    if data_inds_config is not None:
        inds_df = pd.read_csv(data_inds_config, header=None)
        n_data_splits = len(np.unique(inds_df.iloc[:,0].values))

        split_inds = inds_df.iloc[:,0].values
        subject_inds = inds_df.iloc[:,1].values

        subject_ids_lists = [subject_inds[np.where(split_inds==n)].tolist() for n in range(n_data_splits)]
        inds_lists = [[ind for i in subject_ids_lists[n] for ind in inds_paired[i]] for n in range(n_data_splits)]

        
    else:
        x = int(0.2*n_unique)
        split_inds = [n_unique - (j+1)*x for j in reversed(range(n_data_splits-1))]
        split_inds = [0] + split_inds + [n_unique]
        inds_lists = [None] * n_data_splits
        
        for n in range(n_data_splits):
            inds_lists[n] = [ind for pair in inds_paired[split_inds[n]:split_inds[n+1]] for ind in pair]


    return inds_lists, n_inputs



def call_dataset(input_data_files:str,
                 aug_config:str=None,
                 data_inds_config:str=None,
                 n_data_splits:int=1,
                 crop_patch_size=None,
                 use_crop_template:bool=False,
                 prob_lr_flip:int=0.5,
                 n_subjects=None,
                 randomize=True,
                 device=None,
                 segment_pituitary:bool=True,
                 segment_pineal:bool=True,
                 apply_postpit_aug:bool=False,
                 apply_pineal_aug:bool=True,
                 apply_robust_normalization:bool=False,
                 has_ground_truth:bool=True,
):

    if segment_pituitary and segment_pineal:
        data_labels = (0, 883, 900, 903, 904)
    elif segment_pituitary and not segment_pineal:
        data_labels = (0, 883, 903, 904)
    elif not segment_pituitary and segment_pineal:
        data_labels = (0, 900)
    
    lut_path='data_utils/pglands.min.ctab'
    
    # Figure out what type of data we have (this is not elegant sorry)
    if 't1t2pd' in input_data_files:
        input_data_types = ['t1', 't2', 'pd']
    elif 't1t2' in input_data_files:
        input_data_types = ['t1', 't2']
    elif 't1pd' in input_data_files:
        input_data_types = ['t1', 'pd']
    else:
        input_data_types = ['t1']
    

    # Initialize data augmentations
    base_augs, full_augs = augmentation_setup(aug_config=aug_config,
                                              data_types=input_data_types,
                                              label_values=data_labels,
                                              crop_patch_size=crop_patch_size,
                                              use_crop_template=use_crop_template,
                                              prob_lr_flip=0.5,
                                              prob_augment_label_intensities=0.5,
                                              lut_path=lut_path,
                                              random_crop=True if aug_config is not None else False,
                                              apply_postpit_aug=apply_postpit_aug,
                                              apply_pineal_aug=apply_pineal_aug,
                                              apply_robust_normalization=apply_robust_normalization,
                                              has_ground_truth=has_ground_truth,
    )

    # Split input data into train/validation/test cohorts (according to n_data_splits)
    inds_lists, n_inputs = get_inds(input_data_files,
                                    data_inds_config,
                                    n_data_splits=n_data_splits,
                                    n_include=n_subjects,
                                    has_ground_truth=has_ground_truth,
                                    has_template_labels=use_crop_template,
    )
    
    # Compile to generate datasets
    datasets = [None] * len(inds_lists)
    for n in range(n_data_splits):
        datasets[n] = PituitaryPinealDataset(data_inds=inds_lists[n],
                                             data_file=input_data_files,
                                             data_labels=data_labels,
                                             base_augmentations=base_augs,
                                             full_augmentations=full_augs,
                                             n_input=n_inputs,
                                             n_class=len(data_labels),
                                             lut_path=lut_path,
                                             use_template_labels=use_crop_template,
                                             has_ground_truth=has_ground_truth,
                                             device=device
        )
        
    return datasets
