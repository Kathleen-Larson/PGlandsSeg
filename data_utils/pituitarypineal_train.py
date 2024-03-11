import os
from glob import glob
from pathlib import Path
import random

import logging
import pandas as pd
import nibabel as nib
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
import freesurfer as fs

from . import transforms as t



def call_freeview(img, seg):
    fv = fs.Freeview()
    fv.vol(img[0,:])
    fv.vol(seg[0,:], colormap='lut')
    fv.show()


class PituitaryPinealDataset(Dataset):
    def __init__(self,
                 data_inds:list,
                 n_input:int=1,
                 n_class:int=1,
                 image_label_list=None,
                 data_dir=None,
                 transform=None,
                 base_augmentation=None,
                 full_augmentation=None,
                 output_aug_param_path:str=None,
                 save_image_label_list=False,
                 make_RAS=False,
                 **kwargs
    ):
        """
        Args:
            image_label_list (string): CSV file with paths of the images and labels.
            data_dir (string): Directory with the data. Should have subfolders for each modality and 'labels'.
        """
        self.n_input = n_input
        self.n_class = n_class
        self.data_inds = data_inds
        self.make_RAS = make_RAS

        self.transform = transform
        self.base_augmentation = base_augmentation
        self.full_augmentation = full_augmentation
        self.output_aug_param_path = output_aug_param_path
            
        if image_label_list is not None:
            if not os.path.isfile(image_label_list):
                raise ValueError(f'File {image_label_list} does not exist')

            # Read the image and label paths from the file
            df = pd.read_csv(image_label_list, header=None)
            self.image_files = df.iloc[data_inds, :-1].values.tolist()
            self.label_files = df.iloc[data_inds, -1].values.tolist()

        elif data_dir is not None:
            data_dir = Path(data_dir)
            
            if not data_dir.exists():
                raise ValueError(f'Data directory {data_dir} does not exist')
                
            img_dir = data_dir / 'images'
            lbl_dir = data_dir / 'labels'
            
            if not img_dir.is_dir():
                raise ValueError(f'Expected image subfolder {img_dir}')
            if not lbl_dir.is_dir(): 
                raise ValueError(f'Expected label subfolder {lbl_dir}')

            img_files = sorted(glob(str(img_dir / '*')))
            lbl_files = sorted(glob(str(lbl_dir / '*')))

            # Check for mismatches
            if len(img_files) != len(lbl_files):
                logging.warning('Number of images and labels do not match!')
                
                # Take intersection by file name
                img_names = {Path(f).stem for f in img_files}
                lbl_names = {Path(f).stem for f in lbl_files}
                intersect = img_names & lbl_names
                
                img_files = [f for f in img_files if Path(f).stem in intersect]
                lbl_files = [f for f in lbl_files if Path(f).stem in intersect]
                
                logging.info(f'Taking intersection of {len(img_files)} image/label pairs')
            
            self.image_files = img_files
            self.label_files = lbl_files

        else:
            raise ValueError('Either data_dir or image_label_list must be provided')


        if save_image_label_list:
            if not image_label_list and save_image_label_list:
                image_label_list = Path(data_dir) / 'image_pairs.csv'
            os.makedirs(Path(image_label_list).parent, exist_ok=True)
            
            if Path(image_label_list).suffix != '.csv':
                raise ValueError('File must be a CSV')
                
            df = pd.DataFrame({'image_path': self.image_files, 'label_path': self.label_files}) 
            df.to_csv(image_label_list, index=False)

        
    def __len__(self):
        return len(self.image_files)


    def __numinput__(self) -> int:
        return self.n_input


    def __numclass__(self) -> int:
        return self.n_class

    
    def _load_volume(self, path, data_type):
        data = nib.funcs.as_closest_canonical(nib.load(path)) if self.make_RAS else nib.load(path)
        vol = data.get_fdata().astype(data_type)
        return vol


    def _save_output(self, img, path, dtype, is_onehot:bool=False):
        aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        header = nib.Nifti1Header()
        img = torch.argmax(img , dim=1)[0,:] if is_onehot else torch.squeeze(img)
        nib.save(nib.Nifti1Image(img.cpu().numpy().astype(dtype), aff, header), path)

    
    def __getitem__(self, idx):
        img_paths = self.image_files[idx]
        label_path = self.label_files[idx]
        
        images = [self._load_volume(Path(img_path), np.float32) for img_path in img_paths]
        label = self._load_volume(Path(label_path), np.float32)

        images = torch.stack([torch.tensor(image) for image in images], dim=0)
        label = torch.tensor(np.expand_dims(label, axis=0))
        
        return images, label, idx



def augmentation_setup(aug_config:str=None,
                       label_values:list[int]=None,
                       crop_patch_size:list[int]=None,
                       prob_lr_flip:int=0.5,
                       **kwargs
):
    X = 3
    if crop_patch_size is not None:
        center_patch = t.GetPatch(patch_size=crop_patch_size, X=X, randomize=False)
        rand_patch = t.GetPatch(patch_size=crop_patch_size, X=X, randomize=True)
    else:
        center_patch = None
        rand_patch = None
        
    flip = t.RandomLRFlip(axis=X-1, chance=prob_lr_flip)
    norm = t.MinMaxNorm()
    onehot = t.AssignOneHotLabels(label_values=label_values, X=X)

        
    if aug_config is not None:
        df = pd.read_table(aug_config, delimiter='=', header=None)

        translation_bounds = df.loc[df.iloc[:,0]=="translation_bounds",1].item()
        rotation_bounds = df.loc[df.iloc[:,0]=="rotation_bounds",1].item()
        shear_bounds = df.loc[df.iloc[:,0]=="shear_bounds",1].item()
        scale_bounds = df.loc[df.iloc[:,0]=="scale_bounds",1].item()
        max_elastic_displacement = df.loc[df.iloc[:,0]=="max_elastic_displacement",1].item()
        n_elastic_control_pts = df.loc[df.iloc[:,0]=="n_elastic_control_pts",1].item()
        n_elastic_steps = df.loc[df.iloc[:,0]=="n_elastic_steps",1].item()
        gamma_lower = df.loc[df.iloc[:,0]=="gamma_lower",1].item()
        gamma_upper = df.loc[df.iloc[:,0]=="gamma_upper",1].item()
        shape = df.loc[df.iloc[:,0]=="shape",1].item()
        v_max = df.loc[df.iloc[:,0]=="v_max",1].item()
        order = df.loc[df.iloc[:,0]=="order",1].item()
        sigma = df.loc[df.iloc[:,0]=="sigma",1].item()
        
        spatial = t.RandomElasticAffineCrop(translation_bounds=translation_bounds,
                                            rotation_bounds=rotation_bounds,
                                            shear_bounds=shear_bounds,
                                            scale_bounds=scale_bounds,
                                            max_elastic_displacement=max_elastic_displacement,
                                            n_elastic_control_pts=int(n_elastic_control_pts),
                                            n_elastic_steps=int(n_elastic_steps),
                                            X=X
        )
        contrast = t.ContrastAugmentation(gamma_std=0.5) #(gamma_range=(gamma_lower, gamma_upper))
        bias = t.BiasField(shape=int(shape), v_max=int(v_max), order=int(order))
        noise = t.GaussianNoise(sigma=sigma)

        full_augmentation = t.Compose([flip, onehot, spatial, rand_patch, bias, noise, norm, contrast])
        #full_augmentation = t.Compose([flip, onehot, spatial, rand_patch, contrast, bias, noise, norm])
        
    else:
        full_augmentation = t.Compose([onehot, center_patch, norm])
        
    base_augmentation = t.Compose([onehot, center_patch, norm])

    return full_augmentation, base_augmentation



def get_inds(data_config:str, n_include:int=None):
    df = pd.read_csv(data_config, header=None)
    uniqueIDs = df.iloc[:,0].unique()
    [n_subjects, n_inputs] = df.shape
    n_unique = uniqueIDs.size

    if n_include is not	None:
        if n_include < n_unique:
            n_unique = n_include
            uniqueIDs = uniqueIDs[0:n_unique]
        else:
            print(f'n_include ({n_include}) >= n_subjects ({n_subjects}) --> using entire dataset')
    
                  
    random.shuffle(uniqueIDs)
    inds_paired = [None] * n_unique
    for i in range(n_unique):
        inds_paired[i] = df.loc[df.iloc[:,0]==uniqueIDs[i]].index.to_list()

    x = int(0.2*n_unique)
    test_inds = [ind for pair in inds_paired[:x] for ind in pair]
    valid_inds = [ind for pair in inds_paired[x:2*x] for ind in pair]
    train_inds = [ind for pair in inds_paired[2*x:] for ind in pair]

    return train_inds, valid_inds, test_inds, n_inputs-1



def call_dataset(data_config:str, aug_config:str=None, crop_patch_size=None, prob_lr_flip:int=0.5, n_subjects=None):
    data_labels = (0, 883, 900, 903, 904)
    transform = None
    full_aug, base_aug = augmentation_setup(aug_config=aug_config,
                                            label_values=data_labels,
                                            crop_patch_size=(160, 160, 160) if crop_patch_size is None else crop_patch_size,
                                            prob_lr_flip=prob_lr_flip,
    )
    train_inds, valid_inds, test_inds, n_inputs = get_inds(data_config, n_include=n_subjects)
    train = PituitaryPinealDataset(data_inds=train_inds,
                                   image_label_list=data_config,
                                   transform=transform,
                                   base_augmentation=base_aug,
                                   full_augmentation=full_aug,
                                   n_input=n_inputs,
                                   n_class=len(data_labels),
    )
    valid = PituitaryPinealDataset(data_inds=valid_inds,
                                   image_label_list=data_config,
                                   transform=transform,
                                   base_augmentation=base_aug,
                                   full_augmentation=base_aug,
                                   n_input=n_inputs,
                                   n_class=len(data_labels),
    )
    test = PituitaryPinealDataset(data_inds=test_inds,
                                  image_label_list=data_config,
                                  transform=transform,
                                  base_augmentation=base_aug,
                                  full_augmentation=base_aug,
                                  n_input=n_inputs,
                                  n_class=len(data_labels),
    )
    return train, valid, test
