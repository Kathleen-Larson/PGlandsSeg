import os
import warnings
import argparse
import logging

import numpy as np
import nibabel as nib
import freesurfer as fs

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint as pl_ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from data_utils.pituitarypineal_crop import call_dataset
from data_utils import transforms as t

import options
import models
from models import unet, metrics, metrics_pl
from models.segment_pl import Segment as segment
import models.loss_functions as loss_fns
#from models.progress import ProgressBar as ProgressBar
#from models import losses



def setup_log(file, str):
    f = open(file, 'w')
    f.write(str + '\n')
    f.close()
    


### Arg parsing
parser = argparse.ArgumentParser()
#parser = pl.Trainer.add_argparse(parser)
parser = options.set_argparse_defs(parser)
parser = options.add_argparse(parser)
pargs = parser.parse_args()



### Torch set-up ###
pl.seed_everything(pargs.seed, workers=True) #####
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')

warnings.filterwarnings('ignore',
                        "Your \\`val_dataloader\\`\\'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test/predict dataloaders.")



### Data set-up ###
data_config = pargs.data_config
aug_config = pargs.aug_config
train_data, valid_data, test_data = call_dataset(data_config=data_config, aug_config=aug_config)

train_loader = DataLoader(train_data,
                          batch_size=pargs.batch_size,
                          shuffle=True,
                          num_workers=pargs.n_workers,
                          pin_memory=True
)
valid_loader = DataLoader(valid_data,
                          batch_size=pargs.batch_size,
                          shuffle=True,
                          num_workers=pargs.n_workers,
                          pin_memory=True
)
test_loader = DataLoader(test_data,
                         batch_size=pargs.batch_size,
                         shuffle=False,
                         num_workers=pargs.n_workers,
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
lr_param = 0.1 #pargs.lr_param
decay = pargs.weight_decay
optimizer = torch.optim.Adam(network.parameters(),
                             lr=lr_start,
                             weight_decay=decay
)
schedule = pargs.lr_scheduler

metrics_train = [models.metrics_pl.__dict__[pargs.metrics_train[i]]() for i in range(len(pargs.metrics_train))]
metrics_test = [models.metrics_pl.__dict__[pargs.metrics_test[i]]() for i in range(len(pargs.metrics_test))]
metrics_valid = [models.metrics_pl.__dict__[pargs.metrics_valid[i]]() for i in range(len(pargs.metrics_valid))]

output_folder = pargs.output_dir
if output_folder is not None and not os.path.exists(output_folder):  os.mkdir(output_folder)

trainee = segment(model=network,
                  optimizer=optimizer,
                  loss=loss_function,
                  train_data=train_data,
                  valid_data=valid_data,
                  test_data=test_data,
                  output_folder=output_folder,
                  seed=pargs.seed,
                  lr_start=lr_start,
                  lr_param=lr_param,
                  train_metrics=metrics_train,
                  valid_metrics=metrics_valid,
                  test_metrics=metrics_test,
                  save_train_output_every=50,
                  save_valid_output_every=50,
                  schedule=schedule,
)


### Set up log files
"""
setup_log(os.path.join(output_folder, "training_loss.txt"), "Epoch AvgLoss " + \
          " ".join(pargs.metrics_train[i] for i in range(len(pargs.metrics_train)))) + \
          " ".join(pargs.metrics_valid[i] for i in range(len(pargs.metrics_valid)))
"""
if output_folder is not None:
    setup_log(os.path.join(output_folder, "testing_loss.txt"), "ImgID Loss Accuracy " + \
              " ".join(pargs.metrics_test[i] for i in range(len(pargs.metrics_test))))


### Run ? ###
print("Train: %d | Valid: %d | Tests: %d" % \
      (len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)))

callbacks = [pl_ModelCheckpoint(monitor='valid_metric_0',
                                mode='max',
                                dirpath=output_folder,
                                filename='best',
                                save_last=True,
                                every_n_epochs=1),
]

trainer = pl.Trainer(accelerator='gpu',
                     callbacks=callbacks,
                     enable_progress_bar=False,
                     devices=1,
                     log_every_n_steps=len(train_loader),
                     max_epochs=max_n_epochs,
                     gradient_clip_val=0.5,
                     gradient_clip_algorithm='value',
                     precision=16,
                     default_root_dir=output_folder,
)

trainer.fit(trainee, train_loader, valid_loader)
trainer.validate(trainee, valid_loader, verbose=False)
trainer.test(trainee, test_loader, verbose=False)


