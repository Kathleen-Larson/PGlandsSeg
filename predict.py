import os, warnings, argparse, logging, random, csv
from os import system
import numpy as np
import nibabel as nib
import freesurfer as fs
import surfa as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader

from scipy import ndimage

import options
import models
from models import unet, loss_functions
from models.segment import Segment as segment
import models.augmentations as aug

from data_utils.dataloader_pseg import call_dataset



def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    os.environ["PYTHONHASHSEED"] = str(seed)



def _convert_labels(self, img, labels=(0, 883, 900, 903, 904)):
    img_convert = img.copy()
    for i in range(len(labels)):
        img_convert[img==i] = labels[i]
        
    return img_convert
    
    

def _config_optimizer(config, network_params):
    optimizerID = config.optimizer

    if optimizerID=="Adam" or optimizerID=="AdamW":
        betas = (config.beta1, config.beta2)
        optimizer = torch.optim.__dict__[optimizerID](params=network_params,
                                                      betas=(config.beta1, config.beta2),
                                                      lr=config.lr_start,
                                                      weight_decay=config.weight_decay
        )
    elif optimizerID=="SGD":
        optimizer = torch.optim.__dict__[optimizerID](params=network_params,
                                                      lr=config.lr_start,
                                                      momentum=config.momentum,
                                                      weight_decay=config.weight_decay,
                                                      dampening=config.dampening
        )
    else:
        raise Exception('invalid optimizer')

    return optimizer


def _config_lr_scheduler(config, optimizer):
    schedulerID = config.lr_scheduler

    if schedulerID=="ConstantLR":
        scheduler = torch.optim.lr_scheduler.__dict__[schedulerID](optimizer,
                                                                   factor=config.lr_param,
                                                                   total_iters=config.max_n_epochs,
        )
    elif schedulerID=="StepLR":
        scheduler = torch.optim.lr_scheduler.__dict__[schedulerID](optimizer,
                                                                   step_size=config.max_n_epochs,
        )
    elif schedulerID=="PolynomialLR":
        scheduler = torch.optim.lr_scheduler.__dict__[schedulerID](optimizer,
                                                                   total_iters=config.max_n_epochs,
                                                                   power=config.lr_param
        )
    elif schedulerID=="ExponentialLR":
        scheduler = torch.optim.lr_scheduler.__dict__[schedulerID](optimizer,
                                                                   gamma=config.lr_param
        )
    else:
        raise Exception("invalid lr_scheduler")

    return scheduler



def _config():
    parser = argparse.ArgumentParser()
    parser = options.add_argparse(parser)
    pargs = parser.parse_args()

    seed = pargs.seed
    _set_seed(seed)
    return pargs



def _predict(x, model):
    with torch.no_grad(): logits = model(x)
    pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
    return pred

        

def main(pargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('medium')

    mode = 'predict'

    
    ## Initialize data
    aug_config = pargs.aug_config
    input_data_files = pargs.input_data_files
    crop_patch_size = pargs.crop_patch_size
    use_crop_template = bool(pargs.use_crop_template)

    segment_pituitary = pargs.segment_pituitary
    segment_pineal = pargs.segment_pineal
    
    dataset = call_dataset(input_data_files=input_data_files,
                           aug_config=None,
                           crop_patch_size=crop_patch_size,
                           use_crop_template=use_crop_template,
                           segment_pituitary=segment_pituitary,
                           segment_pineal=segment_pineal,
                           has_ground_truth=True if mode == 'train' else False,
                           device=device,
    )[0]
    loader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)

    ## Model set-up
    network = models.unet.__dict__[pargs.network](in_channels=dataset.__numinput__(),
                                                  out_channels=dataset.__numclass__(),
    ).to(device)
    loss_fns = [loss_functions.__dict__[loss_fn] for loss_fn in pargs.loss_fns]
    optimizers = [_config_optimizer(pargs, network.parameters()) for i in range(len(loss_fns))]
    lr_schedulers = [_config_lr_scheduler(pargs, optim) for optim in optimizers]
    
    
    ### Load pre-trained model parameters ###
    train_checkpoint_path = pargs.model_state_path
    if train_checkpoint_path is None:
        raise Exception('model_state_path must be provided')
    else:
        train_checkpoint = torch.load(train_checkpoint_path)
        network.load_state_dict(train_checkpoint["model_state"])
        optimizers[0].load_state_dict(train_checkpoint["optimizer_state"])
        model_epoch = train_checkpoint["epoch"]

        
    ### Parse everything into trainer and run
    output_folder = pargs.output_dir
    start_epoch = 0 if train_checkpoint_path is None else train_checkpoint["epoch"]
    steps_per_epoch = pargs.steps_per_epoch
    max_n_epochs = pargs.max_n_epochs
    start_aug_on = pargs.start_aug_on if pargs.start_aug_on is not None else max_n_epochs
    switch_loss_on = 0
    
    trainer = segment(model=network,
                      optimizer=optimizers[0],
                      scheduler=lr_schedulers[0],
                      loss_functions=loss_fns,
                      start_epoch=start_epoch,
                      max_n_epochs=max_n_epochs,
                      start_full_aug_on=0,
                      steps_per_epoch=steps_per_epoch,
                      switch_loss_on=switch_loss_on,
                      output_folder=output_folder,
                      device=device,
    )

    save_dir = '_'.join(['model_outputs__epoch', str(model_epoch)])
    trainer.test(loader=loader,
                 save_dir=save_dir,
                 mode=mode,
                 save_output=True,
                 write_inputs=False,
                 write_targets=True,
                 write_posteriors=False)
    
            
##########################
if __name__ == "__main__":
    main(_config())
