import os
from os import system
import sys
import warnings
import argparse
import logging
import random
import time
import yaml
import numpy as np
import surfa as sf

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, RandomSampler

from . import loss_functions utils
import augmentations as aug
from dataset import _config_datasets
from segmenter import PGlandsSegmenter
from unet import UNet3D


#------------------------------------------------------------------------------

def main(pargs):
    # Parse commandline args
    use_cuda = pargs.use_cuda
    resume_training = pargs.resume

    if not 'config' in sys.argv:
        print('No .yaml config file supplied, using default '
              'configs/train.yaml')
    config = yaml.safe_load(open(pargs.config))

    # Set up device
    device = 'cpu'
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print('CUDA is unavailable, running w/ cpu')

    utils.set_seeds(config['seed'])
    torch.set_float32_matmul_precision('medium')
    
    
    # Build data loaders
    dataset = _config_datasets(data_config=config['dataset'],
                               aug_config=config['augmentations'],
                               log_dir=config['training']['output_dir'],
                               infer_only=infer_only,
                               device=device,
    )['test']
    loader = DataLoader(dataset, **config['dataloader'])
        
    # Initialize model + optimizer
    network = UNet3D(in_channels=datasets['test'].__numinput__(),
                     out_channels=datasets['test'].__numclass__(),
                     **config['network']
    ).to(device)
    optimizer = _config_optimizer(network.parameters(), **config['optimizer'])

    # Configure segmenter
    segmenter = PGlandsSegmenter(model=network,
                                 optimizer=optimizer,
                                 resume=resume_training,
                                 infer_only=infer_only,
                                 device=device,
                                 n_train_samples=n_train_samples,
                                 **config['training']
    )
        
    # Run
    print(f'Predicting labels for {len(datasets["test"])} input images')
    segmenter._predict(
        loader=loaders['test'], outbase='pglands',
        save_outputs=True, write_posteriors=True,
    )
        

#------------------------------------------------------------------------------
    
def _config_optimizer(network_params, **config):
    optimizerID = config['_class']
    lr_start = config['lr_start']
    weight_decay = config['weight_decay']
    
    if 'Adam' in optimizerID:
        betas = tuple(config['betas'])        
        optimizer = eval(optimizerID)(params=network_params,
                                      betas=betas,
                                      lr=lr_start,
                                      weight_decay=weight_decay,
        )
    elif 'SGD' in optimizerID:
        dampening = config['dampening']
        momentum = config['momentum']
        optimizer = eval(optimizerID)(params=network_params,
                                      lr=lr_start,
                                      momentum=momentum,
                                      weight_decay=weight_decay,
                                      dampening=dampening,
        )
    else:
        raise Exception('invalid optimizer')

    return optimizer


#------------------------------------------------------------------------------

if __name__ == "__main__":
    main(utils.parse_args())
