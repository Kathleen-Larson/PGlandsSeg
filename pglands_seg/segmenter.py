import os, time, shutil
import pathlib as Path
import numpy as np
import freesurfer as fs
import surfa as sf

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import utils, models


#------------------------------------------------------------------------------

class PGlandsSegmenter:
    def __init__(self,
                 model,
                 loss_funcs,
                 optimizer,
                 checkpoint_path:str=None,
                 device:str=None,
                 infer_only:bool=False,
                 max_n_epochs:int=None,
                 max_n_steps:int=None,
                 n_train_samples:int=None,
                 output_dir:str=None,
                 save_outputs_every:int=None,
                 start_aug_on:int=None,
                 steps_per_epoch:int=None,
                 switch_loss_on:int=None,
                 **kwargs
    ):
        # Required inputs
        self.model = model
        self.loss_funcs = [eval(func) for func in loss_funcs]
        self.optimizer = optimizer
        
        # Parse config args
        self.checkpoint_path = checkpoint_path
        self.device = 'cpu' if device is None else device
        self.infer_only = infer_only
        self.max_n_epochs = max_n_epochs
        self.max_n_steps = max_n_steps
        self.output_dir = output_dir
        self.save_outputs_every = save_outputs_every
        self.start_aug_on = start_aug_on
        self.steps_per_epoch = steps_per_epoch
        self.switch_loss_on = switch_loss_on

        # Configure training specific args
        if not self.infer_only:
            # Number of steps per epoch
            if self.steps_per_epoch is None and n_train_samples is None:
                utils.fatal('Error initializing PGlandsSegmenter: must '
                            'specify either steps_per_epoch and/or '
                            'n_train_samples.')
            if self.steps_per_epoch is None:
                self.steps_per_epoch = n_train_samples
            elif n_train_samples is not None and \
                 self.steps_per_epoch != n_train_samples:
                print('Mismatch between steps_per_epoch '
                      '({self.steps_per_epoch})  and n_train_samples '
                      '({n_train_samples}), using n_train_samples')
                self.steps_per_epoch = n_train_samples

            # Maximum number of steps
            if self.max_n_epochs is None and self.max_n_steps is None:
                utils.fatal('Error initializing PGlandsSegmenter: must '
                            'specify either max_n_steps or max_n_epochs')
            if self.max_n_epochs is None:
                self.max_n_epochs = self.max_n_steps // self.steps_per_epoch

            # Frequency of model outputs
            self.print_loss_every = 1 if self.steps_per_epoch < 10 \
                else self.steps_per_epoch // 10
            self.save_model_every = 1 if self.max_n_epochs < 10 \
                else self.max_n_epochs // 10
        
        # Initialize at step/epoch 0
        self.current_step = 0
        self.train_loss = {'last': None, 'best': None}
        self.valid_loss = {'last': None, 'best': None}
        self.loss_func = self.loss_funcs[0] \
            if isinstance(self.loss_funcs, list) else self.loss_funcs
        self.start_epoch = 0
        
        # Load state (if necessary)
        if self.checkpoint_path is not None:
            if not os.path.isfile(self.checkpoint_path):
                utils.fatal(f'Training checkpoint {self.checkpoint_path} not '
                            'found')
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.start_epoch = checkpoint['epoch']
            self.train_loss = checkpoint['train_loss']
            self.valid_loss = checkpoint['valid_loss']

        # Set up output logs
        if self.output_dir is not None:
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)
                
            self.model_output = os.path.join(self.output_dir, 'model')
            self.test_log = os.path.join(self.output_dir, 'test_log.txt')
            utils.init_text_file(self.test_log, 'FileId Loss')
            
            if not infer_only:
                self.train_log = os.path.join(self.output_dir, 'train_log.txt')
                utils.init_text_file(self.train_log, 'Epoch Loss')
                self.valid_log = os.path.join(self.output_dir, 'valid_log.txt')
                utils.init_text_file(self.valid_log, 'Epoch Loss')
            else:
                self.train_log = None
                self.valid_log = None
        else:
            self.model_output = None
            self.test_log = None
            self.train_log = None
            self.valid_log = None


    #--------------------------------------------------------------------------
    
    def _train(self, loader):
        """
        Training loop
        """
        N = len(loader.dataset)
        loss_avg = 0.

        self.model.train()
        self.model.zero_grad()

        # Iterate through samples
        for data, idx in loader:
            X, y, _ = loader.dataset.augmentations(
                [x.to(self.device) for x in data]
            )
            logits = self.model(X)
            loss = self.loss_func(logits, y)
            loss_avg += loss.item()
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.current_step += 1
            if self.current_step % (self.print_loss_every) == 0:
                print(f'Epoch={self.current_epoch}, '
                      + f'step={self.current_step}: '
                      + f'loss={loss.item():>.4f}')

        # Update loss/logs
        self.train_loss['last'] = loss_avg / N
        if self.train_loss['last'] < self.train_loss['best']:
            self.train_loss['best'] = train_loss['last']
        
        if self.train_log is not None:
            with open(self.train_log, 'a') as f:
                f.write(f'{self.current_epoch:>5}')
                f.write(f'{self.train_loss["last"]:>9.4f}')
                f.write(f'\n')

                
    #--------------------------------------------------------------------------
    
    def _predict(self, loader,
                 loss_type:str='test',
                 save_dir:str=None,
                 outbase:str='pglands',
                 save_outputs:bool=False,
                 write_inputs:bool=True,
                 write_targets:bool=True,
                 write_posteriors:bool=True,
    ):
        """
        Inference (validation/testing) loop
        """

        N = len(loader.dataset)
        loss_avg = None if loader.dataset.label_files is None else 0.

        if write_targets and loader.dataset.label_files is None:
            print('Cannot write targets because dataset does not contain '
                  'ground truth... check your data config files if this is '
                  'unexpected.')
            write_targets = False
        
        self.model.eval()
        self.model.zero_grad()
        
        for data, idx in loader:
            # Load data
            if loader.dataset.label_files is None:
                X, _ = loader.dataset.augmentations(
                    [x.to(self.device) for x in data]
                )
                y = None
            else:
                X, y, _ = loader.dataset.augmentations(
                    [x.to(self.device) for x in data]
                )

            # Run inference
            with torch.no_grad():
                logits = self.model(X)

            # Calculate loss
            if y is not None:
                loss = self.loss_func(logits, y).item()
                loss_avg += loss
                if loss_type == 'test':                    
                    with open(self.test_log, 'a') as f:
                        f.write(f'{loader.dataset.outbases[idx]} '
                                f'{loss:>.4f}\n')
                        
            # Write outputs?
            if save_outputs:
                posts = F.softmax(logits, dim=1).squeeze()
                seg = torch.argmax(posts, dim=0)
                
                outdict = {'Input': X if write_inputs else None,
                           'Target': y if write_targets else None,
                           'Posteriors': posts if write_posteriors else None,
                           'Output': seg
                }                
                _save_model_outputs(data_dict=outdict,
                                    dataloader_idx=idx,
                                    save_dir=save_dir,
                                    outbase=outbase
                )

        # Update loss/logs
        if loss_type == 'valid':
            self.valid_loss['last'] = loss_avg / N
            if self.valid_loss['last'] < self.valid__loss['best']:
                self.valid_loss['best'] = valid__loss['last']

            if self.valid_log is not None:
                with open(self.train_log, 'a') as f:
                    f.write(f'{self.current_epoch:>5}')
                    f.write(f'{self.valid_loss["last"]:>9.4f}')
                    f.write(f'\n')
            

    #--------------------------------------------------------------------------

    ##
    def _epoch_end(self, output_model:bool=False):
        """
        Runs at the end of each training epoch (after validation steps)
        """
        self.current_step = 0
        self.current_epoch += 1
        
        # Save model
        if (self.model_output is not None
            and (self.current_epoch + 1) % self.save_model_every == 0
        ):
            torch.save({'epoch': self.current_epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict()},
                       '_'.join([self.model_output, 'last.pth'])
            )

        # Switch loss?
        if self.current_epoch == self.switch_loss_on:
            self.loss_f = self.loss_funcs[1]
            print(f'Switching from {self.loss_funcs[0].__name__} to '
                  f'{self.loss_funcs[1].__name__} after '
                  f'{self.current_epoch} epochs.')
            


    ##
    def _save_model_outputs(data_dict:dict,
                            dataloader_idx:int,
                            save_dir:str=None,
                            outbase:str='pglands',
    ):        
        save_dir = os.path.join(
            self.output_dir, loader.dataset.outbases[idx]
        ) if save_dir is None else os.path.join(
            self.output_dir, save_dir, loader.dataset.outbases[idx]
        )
        fbase = os.path.join(save_dir, loader.dataset.outbases[idx])
            
        if 'Input' in data_dict:
            [loader.dataset._save_volume(
                img=data_dict['Input'], idx=idx, conform=True,
                path='.'.join([base, 'input', 'mgz']),
            ) for base in fbase.split('+')]
        
        if 'Target' in data_dict:
            loader.dataset._save_volume(
                img=data_dict['Target'], idx=idx, is_labels=True, conform=True,
                path='.'.join([fbase, 'target', 'mgz']),
            )
            
        if 'Posteriors' in data_dict:
            [loader.dataset._save_volume(
                img=data_dict['Posteriors'][n, ...], idx=idx, conform=True,
                path='.'.join([base, loader.dataset.lut[val].name, 'mgz']),
            ) for n, val in enumerate(loader.dataset.lut)]

        if 'Output' in data_dict:
            loader.dataset._save_volume(
                img=data_dict['Output'], idx=idx, is_labels=True, conform=True,
                path='.'.join([fbase, outbase, 'mgz']),
            )


