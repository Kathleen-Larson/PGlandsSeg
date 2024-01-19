import os, warnings, argparse, logging, random
import numpy as np
import nibabel as nib
import freesurfer as fs

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from data_utils.pituitarypineal_crop import call_dataset
from data_utils import transforms as t

import options
import models
from models import unet, metrics
from models.segment import Segment as segment
import models.loss_functions as loss_fns



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
    schedulerID = config.lr_scheduler

    lr_start = config.lr_start
    dampening = config.dampening
    weight_decay = config.weight_decay

    if optimizerID=="Adam" or optimizerID=="AdamW":
        betas = (config.beta1, config.beta2)
        optimizer = torch.optim.__dict__[optimizerID](params=network_params,
                                                      betas=betas,
                                                      lr=lr_start,
                                                      weight_decay=weight_decay
        )
    elif optimizerID=="SGD":
        momentum = config.momentum
        optimizer = torch.optim.__dict__[optimizerID](params=network_params,
                                                      lr=lr_start,
                                                      momentum=momentum,
                                                      weight_decay=weight_decay,
                                                      dampening=dampening
        )
    else:
        raise Exception('invalid optimizer')

    if schedulerID=="StepLR":
        scheduler = torch.optim.lr_scheduler.__dict__[schedulerID](optimizer, step_size=5)
    elif schedulerID=="LinearLR":
        scheduler = torch.optim.lr_scheduler.__dict__[schedulerID](optimizer)
    elif schedulerID=="PolynomialLR":
        scheduler = torch.optim.lr_scheduler.__dict__[schedulerID](optimizer, total_iters=-1, power=0.1)
    elif schedulerID=="ExponentialLR":
        scheduler = torch.optim.lr_scheduler.__dict__[schedulerID](optimizer, gamma=0.9)
    else:
        raise Exception('invalid LR scheduler')

    return optimizer, scheduler



def _config():
    parser = argparse.ArgumentParser()
    parser = options.add_argparse(parser)
    pargs = parser.parse_args()

    seed = pargs.seed
    _set_seed(seed)

    return pargs



def main(pargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('medium')
    
    ### Data set-up ###
    aug_config = pargs.aug_config
    batch_size = pargs.batch_size
    data_config = pargs.data_config
    n_workers = pargs.n_workers

    train_data, valid_data, test_data = call_dataset(data_config=data_config, aug_config=aug_config)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=n_workers,
                              pin_memory=True
    )
    valid_loader = DataLoader(valid_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=n_workers,
                              pin_memory=True
    )
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=n_workers,
                             pin_memory=True
    )
    
    
    ### Model set-up ###
    max_n_epochs = pargs.max_n_epochs
    network = models.unet.__dict__[pargs.network](in_channels=train_data.__numinput__(),
                                                  out_channels=train_data.__numclass__(),
    ).to(device)

    ### Optimizer set-up ###
    loss_function =  loss_fns.__dict__[pargs.loss]
    #optimizer, lr_scheduler = _config_optimizer(pargs, network.parameters())

    lr_start = pargs.lr_start
    #lr_param = 0.1 #args.lr_param
    decay = pargs.weight_decay
    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=lr_start,
                                 weight_decay=decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,
                                                       total_iters=max_n_epochs,
                                                       factor=0.5
    )

    ## Metrics set-up ###
    metrics_train = [models.metrics.__dict__[pargs.metrics_train[i]] \
                     for i in range(len(pargs.metrics_train))]
    metrics_test = [models.metrics.__dict__[pargs.metrics_test[i]] \
                    for i in range(len(pargs.metrics_test))]
    metrics_valid = [models.metrics.__dict__[pargs.metrics_valid[i]] \
                     for i in range(len(pargs.metrics_valid))]


    ### Logging set-up ###
    output_folder = pargs.output_dir
    if output_folder is not None:
        if not os.path.exists(output_folder):  os.mkdir(output_folder)
        _setup_log(os.path.join(output_folder, "training_log.txt"), "Epoch Loss " + \
                   " ".join(pargs.metrics_train[i] for i in range(len(pargs.metrics_train))))
        _setup_log(os.path.join(output_folder, "validation_log.txt"), "Epoch Loss " + \
                   " ".join(pargs.metrics_valid[i] for i in range(len(pargs.metrics_valid))))
        _setup_log(os.path.join(output_folder, "testing_log.txt"), "ImgID Loss " + \
                   " ".join(pargs.metrics_test[i] for i in range(len(pargs.metrics_test))))
        
        
    ### Parse everything into trainer ###
    save_train_output_every=1
    save_valid_output_every=1
    
    trainer = segment(model=network,
                      optimizer=optimizer,
                      scheduler=lr_scheduler,
                      loss=loss_function,
                      start_full_aug_on=50,
                      output_folder=output_folder,
                      device=device,
    )
        

    ### Run ###
    print("Train: %d | Valid: %d | Tests: %d" % \
          (len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)))
    
    n_epochs = max_n_epochs
    for epoch in range(n_epochs):
        trainer.train(loader=train_loader,
                      metrics_list=metrics_train,
                      epoch=epoch,
                      save_output=False
        )
        trainer.validate(loader=valid_loader,
                         metrics_list=metrics_valid,
                         epoch=epoch,
                         save_output=False
        )
        
    trainer.test(loader=test_loader,
                 metrics_list=metrics_test,
                 save_output=False
    )
    
    

##########################
if __name__ == "__main__":
    main(_config())
