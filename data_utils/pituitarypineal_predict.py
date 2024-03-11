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

from . import transforms_singleinput as t



def call_freeview(img, seg):
    fv = fs.Freeview()
    fv.vol(img[0,:])
    fv.vol(seg[0,:], colormap='lut')
    fv.show()


class Dataset_Predict(Dataset):
    def __init__(self,
                 n_input:int=1,
                 n_class:int=1,
                 data_config=None,
                 data_dir=None,
                 transform=None,
                 augmentation=None,
                 make_RAS=False,
                 **kwargs
    ):
        self.n_class = n_class
        self.make_RAS = make_RAS
        self.transform = transform
        self.augmentation = augmentation
            
        if data_config is not None:
            if not os.path.isfile(data_config):
                raise ValueError(f'File {data_config} does not exist')

            df = pd.read_csv(data_config, header=None)
            self.image_files = df.values.tolist()
            self.n_input = df.shape[1]

        else:
            raise ValueError('image_list must be provided')

        
    def __len__(self):
        return len(self.image_files)


    def __numinput__(self) -> int:
        return self.n_input


    def __numclass__(self) -> int:
        return self.n_class

    
    def _load_volume(self, path, data_type, make_RAS=False):
        data = nib.funcs.as_closest_canonical(nib.load(path)) if make_RAS else nib.load(path)
        vol = data.get_fdata().astype(data_type)
        return vol


    def _save_output(self, img, path, dtype, is_onehot:bool=False):
        aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        header = nib.Nifti1Header()
        img = torch.argmax(img , dim=1)[0,:] if is_onehot else torch.squeeze(img)
        nib.save(nib.Nifti1Image(img.cpu().numpy().astype(dtype), aff, header), path)

    
    def __getitem__(self, idx, gpu=True, cpu=False):
        img_paths = self.image_files[idx]
        images = [self._load_volume(Path(img_path), np.float32) for img_path in img_paths]
        images = torch.stack([torch.tensor(image) for image in images], dim=0)

        return images, idx



def augmentation_setup(crop_patch_size:list[int]=None,
                       reference_train_image_path=None,
                       reference_test_image_path=None
):
    X = 3
    breakpoint()
    if reference_train_image_path is not None and reference_test_image_path is not None:
        reference_train_image = nib.load(reference_train_image_path)
        reference_test_image = nib.load(reference_test_image_path)
        resample = t.ResampleImage(input_shape=reference_test_image.header['dims'][:-1],
                                   input_res=reference_test_image.header['delta'],
                                   output_shape=reference_train_image.header['dims'][:-1],
                                   output_res=reference_train_image.header['delta'],
        )
    else:
        resample = None

    center_patch = t.GetPatch(patch_size=crop_patch_size, X=X, randomize=False) if crop_patch_size is not None else None
    norm = t.MinMaxNorm()
    augmentation = t.Compose([resample, center_patch, norm])

    return augmentation



def call_dataset(data_config:str, aug_config:str=None, crop_patch_size=None, n_subjects=None):
    data_labels = (0, 883, 900, 903, 904)

    reference_train_image_path = 'data/input/fsm/intensity/fsm004.intensity.mgz'
    reference_test_image_path = 'data/input/miriad/188_01_MR_1.mgz'
    
    aug = augmentation_setup(crop_patch_size=(160, 160, 160) if crop_patch_size is None else crop_patch_size,
                             reference_train_image_path=reference_train_image_path,
                             reference_test_image_path=reference_test_image_path
    )
    dataset = Dataset_Predict(data_config=data_config,
                              n_class=len(data_labels),
                              transform=None,
                              augmentation=aug
    )
    
    return dataset
