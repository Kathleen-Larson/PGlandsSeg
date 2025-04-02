import os
from pathlib import Path
import random

import logging
import pandas as pd
import numpy as np
import surfa as sf
import torch

from scipy import ndimage

from torch.utils.data import Dataset
from torchvision import transforms

import augmentations as aug
import utils


#------------------------------------------------------------------------------

class PGlandsDataset(Dataset):
    def __init__(self,
                 image_files:list=None,
                 label_files:list=None,
                 template_files:list=None,
                 aug_config:dict=None,
                 aug_type:str=None,
                 lut=None,
                 device=None,
                 n_inputs:int=1,
                 n_class:int=None,
                 infer_only:bool=False,
    ):
        # Parse inputs
        if lut is None and n_class is None:
            utils.fatal('Error in PGlandsDataset: must either input a label '
                        'lut or n_class.')
        self.n_input = n_inputs
        self.n_class = n_class if n_class is not None else len(lut)
        self.lut = lut
        self.device = 'cpu' if device is None else device

        # Store filenames
        self.image_files = image_files
        self.label_files = label_files
        self.template_files = template_files
        self.outbases = ['+'.join([Path(fname).stem for fname in flist])
                         for flist in self.image_files]
        
        # Set up augmentations
        aug_list = [
            None if infer_only
            else aug.AssignOneHotLabels(label_values=[x for x in lut])
        ]
        aug_list += [getattr(aug, func)(**aug_config[aug_type][func])
                     for func in aug_config['_transform_order']
                     if func in aug_config[aug_type]
        ]
        self.augmentations = aug.ComposeTransforms(aug_list)
            
        # Store native image geometries and crop windows
        use_template = True if self.template_files is not None else False
        ref_image_list = self.template_files if use_template \
            else self.image_files
        self.geoms = [[None, None]] * len(self.image_files)
        self.crop_bounds = [None] * len(self.image_files)

        """
        for idx, path in enumerate(ref_image_list):
            ref_path = path[0] if isinstance(path, list) else path
            is_int = True if use_template else False
            img, geom0, geom1 = utils.load_volume(
                path=ref_path, is_int=is_int, to_tensor=False,
                return_geoms=True,
            )
            self.geoms[idx] = [geom0, geom1]
            self.crop_bounds[idx] = None if not use_template \
                else utils.get_crop_window(img)
        """


    def __len__(self):
        return len(self.image_files)


    def __numinput__(self) -> int:
        return self.n_input


    def __numclass__(self) -> int:
        return self.n_class
    

    def _create_file_id_log(self, logname, flist):
        if len(np.array(flist).shape) > 1:
            flist = np.array(flist)[:, 0].tolist()
        fids = sorted(set([os.path.basename(fname) for fname in flist]))
        with open(logname, 'w') as f:
            for fid in fids:
                f.write(fid + '\n')


    def _save_volume(self, img, path, idx,
                     is_labels:bool=False,
                     conform:bool=True
    ):
        utils.save_volume(
            img, path, crop_bounds=self.crop_bounds[idx],
            input_geom=self.geoms[idx][0], conform_geom=self.geoms[idx][1],
            label_lut=self.lut, is_labels=is_labels
        )
    

    def __getitem__(self, idx):
        data = [torch.stack([utils.load_volume(path)
                             for path in self.image_files[idx]], dim=0)]
        if self.label_files is not None:
            data += [utils.load_volume(self.label_files[idx], is_int=True
            ).unsqueeze(0)]
        if self.template_files is not None:
            data += [utils.load_volume(self.template_files[idx], is_int=True
            ).unsqueeze(0)]
        return data, idx


#------------------------------------------------------------------------------

def _config_datasets(data_config:dict, aug_config:dict, infer_only:bool=False,
                     device=None, log_dir:str=None
):
    """
    Configures datasets divided into a specified number of cohorts (e.g. 
    train, test, valid). Outputs all dataset within a single dictionary.
    """

    # Load label lut
    if not os.path.isfile(data_config['lut']):
        utils.fatal(f'{data_config["lut"]} does not exist')
    lut = sf.load_label_lookup(data_config['lut'])
    
    # Load input data config
    if not os.path.isfile(data_config['input_data_config']):
        utils.fatal(f'{data_config["input_data_config"]} does not exist')
    df = pd.read_csv(data_config['input_data_config'])
    cols = df.columns.tolist()
    
    # Parse config file headers
    if not 'InputT1' in cols:
        utils.fatal('Input data config must be a .csv file with a column '
                    'titled "InputT1" containing paths to T1 images for '
                    'segmentation.')
    if not 'GroundTruth' in cols and not infer_only:
        utils.fatal('Input data config  must be a .csv file with a column '
                    'titled "GroundTruth" containing paths to ground '
                    'truth label maps (unless infer_only=True).')
    if not 'LabelTemplate' in cols and \
       not aug_config['CropPatch']['center_crop']:
        utils.fatal('Input data config must be a .csv file with a column '
                    'titled "LabelTemplate" containing paths to label '
                    'templates in mni152 space (unless '
                    'aug_config["CropPatch"]["center_crop"] is true).')
    if not 'SplitId' in cols and not data_config['randomize']:
        utils.fatal('Input data config must be a .csv file with a column '
                    'titled "LabelTemplate" containing paths to label '
                    'templates in mni152 space (unless '
                    'data_config["randomize"] is True).')
        
    # Parse file names from    
    input_image_idxs = ['Input' in col for col in cols]
    image_files = df.iloc[:, input_image_idxs].values
    label_files = df['GroundTruth'].values if 'GroundTruth' in cols else None
    template_files = None if aug_config['full']['CropPatch']['center_crop'] \
        else df['LabelTemplate'].values
    data_split_ids = None if data_config['randomize'] \
        else df['SplitId'].values
    
    [n_subjects, n_inputs] = df.shape
    if label_files is not None: n_inputs -= 1
    if template_files is not None: n_inputs -= 1
    if data_split_ids is not None: n_inputs -= 1
    
    # Get list of unique input images (some may have multiple label maps)
    ref_file_list = np.array(image_files).squeeze()
    unique_refs = sorted(set(ref_file_list),
                         key=ref_file_list.tolist().index)
    n_unique = len(unique_refs)
    
    # Sort subjects into separate cohorts (e.g. train/valid/test)
    n_splits = data_config['n_splits']
    if len(np.unique(data_split_ids)) != n_splits:
        print(f'Mismatch between data_config["n_splits"] and number of '
              'SplitIds in input data config.. using n_split = '
              'len(set(data_split_ids)) = {len(set(data_split_ids))}')
    if infer_only:
        n_splits = 1
        data_split_names = ['test']
        idxs_lists = [np.arange(0, image_files.shape[0]).tolist()]
    else:
        if data_split_ids is None:
            random.shuffle(unique_refs)
            data_split_names = ['train', 'test', 'valid'][:n_splits]
            x = int(0.2*n_unique)
            split_idxs = [0] + [
                n_unique - (j+1) * x for j in reversed(range(n_splits-1))
            ] + [n_unique]
            idx_groups = [np.where(ref_file_list == ref)[0].tolist()
                          for ref in unique_refs]
            idxs_lists = [
                [idx for idxs in idx_groups[split_idxs[n]:split_idxs[n+1]]
                 for idx in idxs] for n in range(n_splits)
            ]
        else:
            data_split_names = np.unique(data_split_ids).tolist()
            idx_groups = [np.where(ref_file_list == ref)[0].tolist()
                          for ref in unique_refs]
            data_split_ids = np.array([data_split_ids[idxs[0]]
                                       for idxs in idx_groups])
            subject_idxs = np.arange(0, n_unique)
            subject_ids_lists = [
                subject_idxs[np.where(data_split_ids == split)].tolist()
                for split in data_split_names
            ]
            idxs_lists = [[idx for i in subject_ids_lists[n]
                           for idx in idx_groups[i]] for n in range(n_splits)]
            
    # Sort file names into each cohort
    datasets_dict = {}
    for n, (idxs, split_name) in enumerate(zip(idxs_lists, data_split_names)):
        ds_config= {}
        ds_config['n_inputs'] = np.sum(input_image_idxs)
        ds_config['n_class'] = len(lut)
        ds_config['image_files'] = image_files[idxs,:].tolist()
        ds_config['label_files'] = None if label_files is None \
            else label_files[idxs].tolist()
        ds_config['template_files'] = None if template_files is None \
            else template_files[idxs].tolist()
        ds_config['aug_type'] = 'full' if split_name == 'train' \
            else 'partial'
        
        ds = PGlandsDataset(aug_config=aug_config, lut=lut, device=device,
                            infer_only=infer_only, **ds_config
        )
        if log_dir is not None:
            ds_log = os.path.join(log_dir, f'{split_name}_file_ids.txt')
            ds._create_file_id_log(ds_log, ds.image_files)
        datasets_dict[split_name] = ds

    return datasets_dict
