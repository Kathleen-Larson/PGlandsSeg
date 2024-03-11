import os, warnings, argparse, logging, random
from os import system
import numpy as np
import nibabel as nib
import freesurfer as fs

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, RandomSampler

from data_utils.pituitarypineal_train import call_dataset
from data_utils import transforms as t

import options
import models
from models import unet, metrics
from models.segment import Segment as segment
import models.loss_functions as loss_functions



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



def _setup_log(file, str):
    f = open(file, 'w')
    f.write(str + '\n')
    f.close()
    


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
        momentum = config.momentum
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
                                                                   step_size=5,
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



def _dataloader_random_sample(pargs, dataset):
    batch_size = pargs.batch_size
    n_workers = pargs.n_workers
    n_samples = pargs.steps_per_epoch

    sampler = RandomSampler(dataset, replacement=True, num_samples=n_samples)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=n_workers, pin_memory=True)
    return dataloader

    


def main(pargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('medium')
    
    
    ### Data set-up ###
    aug_config = pargs.aug_config
    data_config = pargs.data_config

    train_data, valid_data, test_data = call_dataset(data_config=data_config,
                                                     aug_config=aug_config,
                                                     crop_patch_size=(160, 160, 160),
                                                     prob_lr_flip=0.5,
    )
    train_loader = _dataloader_random_sample(pargs, train_data)
    valid_loader = _dataloader_random_sample(pargs, valid_data)
    test_loader = _dataloader_random_sample(pargs, test_data)
    
    
    ### Model set-up ###
    network = models.unet.__dict__[pargs.network](in_channels=train_data.__numinput__(),
                                                  out_channels=train_data.__numclass__(),
    ).to(device)
    loss_fns = [loss_functions.__dict__[loss_fn] for loss_fn in pargs.loss_fns]
    optimizers = [_config_optimizer(pargs, network.parameters()) for i in range(len(loss_fns))]
    lr_schedulers = [_config_lr_scheduler(pargs, optim) for optim in optimizers]

    
    ### Load pre-trained model parameters ###
    train_checkpoint_path = pargs.model_state_path
    pretrain_model_path = pargs.pretrain_model_path
    if train_checkpoint_path is None:
        if pretrain_model_path is not None:
            pretrain_model = torch.load(pretrain_model_path)
            network.load_state_dict(pretrain_model["model_state"])
            #optimizers[0].load_state_dict(pretrained_checkpoint['optimizer_state'])
    else:
        train_checkpoint = torch.load(train_checkpoint_path)
        network.load_state_dict(train_checkpoint["model_state"])
        optimizers[0].load_state_dict(train_checkpoint["optimizer_state"])


    ## Metrics set-up ###
    metrics_train = [models.metrics.__dict__[pargs.metrics_train[i]] \
                     for i in range(len(pargs.metrics_train))]
    metrics_test = [models.metrics.__dict__[pargs.metrics_test[i]] \
                    for i in range(len(pargs.metrics_test))]
    metrics_valid = [models.metrics.__dict__[pargs.metrics_valid[i]] \
                     for i in range(len(pargs.metrics_valid))]


    ### Parse everything into trainer ###
    output_folder = pargs.output_dir
    start_epoch = 0 if train_checkpoint_path is None else train_checkpoint["epoch"]
    steps_per_epoch = pargs.steps_per_epoch
    max_n_epochs = pargs.max_n_epochs
    start_aug_on = pargs.start_aug_on if pargs.start_aug_on is not None else max_n_epochs
    switch_loss_on = pargs.switch_loss_on
    save_train_output_every=1
    save_valid_output_every=1
    
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
                      metrics_train=metrics_train,
                      metrics_valid=metrics_valid,
                      metrics_test=metrics_test
    )
    

    ### Logging set-up ###
    if output_folder is not None and train_checkpoint_path is None:
        if not os.path.exists(output_folder):  os.mkdir(output_folder)
        _setup_log(os.path.join(output_folder, "training_log.txt"), "Epoch TrainLoss LearningRate " + \
                   " ".join(pargs.metrics_train[i] for i in range(len(pargs.metrics_train))))
        _setup_log(os.path.join(output_folder, "validation_log.txt"), "Epoch Loss " + \
                   " ".join(pargs.metrics_valid[i] for i in range(len(pargs.metrics_valid))))
        _setup_log(os.path.join(output_folder, "testing_log.txt"), "ImgID Loss " + \
                   " ".join(pargs.metrics_test[i] for i in range(len(pargs.metrics_test))))
        
        param_output_file=os.path.join(output_folder, "training_params.txt")
        f = open(param_output_file, 'w')
        
        f.write('--------------------------------------\n')
        f.write('Full data augmentation set:\n')        
        f.write('--------------------------------------\n')
        for t in train_data.full_augmentation.transforms:
            f.write(f'{t.__class__.__name__}:\n')
            for k in t.__dict__:
                if k != 'X':  f.write(f'    {k}={t.__dict__[k]}\n')
        f.write('\n')
        
        f.write('--------------------------------------\n')
        f.write('Training parameters:\n')
        f.write('--------------------------------------\n')
        for k in pargs.__dict__:
            f.write(f'{k}={pargs.__dict__[k]}\n')
        f.write('\n')
        
        f.close()
        with open(param_output_file, 'r') as f:
            print(f.read())
        
                
    ### Run ###
    print('--------------------------------------')
    print("Train: %d | Valid: %d | Tests: %d" % \
          (len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)))
    print('--------------------------------------')

    for epoch in range(start_epoch, max_n_epochs):
        trainer.train(loader=train_loader, save_output=False)
        trainer.validate(loader=valid_loader, save_output=False)
        trainer._epoch_end()
    trainer.test(loader=test_loader, save_output=True)
    
    

##########################
if __name__ == "__main__":
    main(_config())
