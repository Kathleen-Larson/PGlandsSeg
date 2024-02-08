import os
import time
import pathlib as Path
import numpy as np
import freesurfer as fs

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class Segment:
    def __init__(self, model, optimizer, scheduler, loss_function, device,
                 output_folder, max_n_epochs, start_epoch=0, start_full_aug_on=None,
                 metrics_train=None, metrics_valid=None, metrics_test=None,
                 use_multiple_optimizers=False, switch_loss_weight_every=20):
        super().__init__()

        self.device = device
        self.model = model

        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.valid_loss = [None, None]

        self.max_n_epochs = max_n_epochs
        self.start_full_aug_on = start_full_aug_on if start_full_aug_on is not None else self.max_n_epochs

        self.current_epoch = start_epoch
        self.current_train_loss_avg = 0
        self.print_training_metrics_on_epoch = 1

        self.metrics_train = metrics_train
        self.metrics_valid = metrics_valid
        self.metrics_test = metrics_test

        self.output_folder = output_folder
        if self.output_folder is not None:
            self.train_output = os.path.join(output_folder, "training_log.txt")
            self.valid_output = os.path.join(output_folder, "validation_log.txt")
            self.test_output = os.path.join(output_folder, "testing_log.txt")
            self.model_output = os.path.join(output_folder, "model")
        else:
            self.train_output = None
            self.valid_output = None
            self.test_output = None
            self.model_output = None

        self.train_example = None
        self.valid_example = None
        
        
            
    ### Training loop
    def train(self, loader, save_output:bool=False):
        N = len(loader.dataset)
        M = len(self.metrics_train)
        loss_avg = 0.0
        metrics = np.zeros((M))
        n_zeroFG = 0
        
        if self.current_epoch < self.start_full_aug_on:
            augmentation = loader.dataset.base_augmentation
        else:
            augmentation = loader.dataset.full_augmentation

        self.model.zero_grad()
        self.model.train()
        
        for X, y, idx in loader:
            X, y = augmentation(X.to(self.device), y.to(self.device))
            
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.loss_fn(logits, y, gpu=True)
                
            loss_avg += loss.item()
            loss.backward()
            self.optimizer.step()

            y_target = torch.argmax(y, dim=1)
            y_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
            if y_pred.max()==0:
                n_zeroFG += 1
            
            if self.metrics_train is not None:
                metrics = [metrics[m] + self.metrics_train[m](y_pred, y_target) for m in range(M)]
            
        if save_output and self.output_folder is not None:
            output_folder_train = os.path.join(self.output_folder, "train_data",
                                               "epoch_" + str(self.current_epoch).zfill(len(str(self.max_n_epochs))))
            self._save_output(output_folder_train, loader.dataset, X, y_target, y_pred, idx)
            
        self.train_example = [X.cpu().detach().numpy().squeeze(),
                              y_target.cpu().detach().numpy().squeeze(),
                              y_pred.cpu().detach().numpy().squeeze(),
                              logits.cpu().detach().numpy().squeeze()]

        self.current_train_loss_avg = loss_avg / N
        metrics = [metrics[m] / N for m in range(M)]

        if n_zeroFG > 0:
            print(f'predictions for {n_zeroFG}/{N} subjects contained 0 FG voxels')

        if self.train_output is not None:
            f = open(self.train_output, 'a')
            f.write(f'{self.current_epoch:5} {self.current_train_loss_avg:>9.4f} {self.scheduler.get_last_lr()[0]:>12.5f}')
            for m in range(M):
                f.write(f' {metrics[m]:>{len(self.metrics_train[m].__name__)}.5f}')
            f.write(f'\n')
            f.close()
            

            
    ### Validation loop
    def validate(self, loader, save_output:bool=False):
        N = len(loader.dataset)
        M = len(self.metrics_valid)
        loss_avg = 0.0
        metrics = np.zeros((M))
        
        if self.current_epoch < self.start_full_aug_on:
            augmentation = loader.dataset.base_augmentation
        else:
            augmentation = loader.dataset.base_augmentation
            
        self.model.eval()

        for X, y, idx in loader:
            X, y = augmentation(X.to(self.device), y.to(self.device))

            with torch.no_grad():
                self.optimizer.zero_grad()
                logits = self.model(X)
                loss = self.loss_fn(logits, y, gpu=True)

            loss_avg += loss.item()
            y_target = torch.argmax(y, dim=1)
            y_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
            
            if self.metrics_valid is not None:
                metrics = [metrics[m] + self.metrics_valid[m](y_pred, y_target) for m in range(M)]

        if save_output and self.output_folder is not None:
            output_folder_valid = os.path.join(self.output_folder, "valid_data",
                                               "epoch_" + str(self.current_epoch).zfill(len(str(self.max_n_epochs))))
            self._save_output(output_folder_valid, loader.dataset, X, y_target, y_pred, idx)
            
        loss_avg = loss_avg / N
        self.valid_loss = [self.valid_loss[1], loss_avg]
        self.valid_example = [X.cpu().detach().numpy(),
                              y_target.cpu().detach().numpy(),
                              y_pred.cpu().detach().numpy(),
                              logits.cpu().detach().numpy()]

        if self.metrics_valid is not None:  metrics = [metrics[m] / N for m in range(M)]

        if self.valid_output is not None:
            f = open(self.valid_output, 'a')
            f.write(f'{self.current_epoch} {loss_avg:>.4f}')
            for m in range(M):
                f.write(f' {metrics[m]:>.5f}')
            f.write(f'\n')
            f.close()

        

    ### Testing loop
    def test(self, loader, save_output:bool=False):
        N = len(loader.dataset)
        M = len(self.metrics_test)

        loss_idx = 0.0
        metrics_idx = np.zeros((M, 1))
        augmentation = loader.dataset.base_augmentation

        self.model.eval()

        for X, y, idx in loader:
            X, y = augmentation(X.to(self.device), y.to(self.device))
            
            with torch.no_grad():
                logits = self.model(X)
                loss = self.loss_fn(logits, y, gpu=True)
                loss_idx = loss.item()

            y_target = torch.argmax(y, dim=1)
            y_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)

            if self.metrics_test is not None:
                metrics_idx = [self.metrics_test[m](y_pred, y_target) for m in range(M)]

            if save_output and self.output_folder is not None:
                self._save_output(os.path.join(self.output_folder, "test_data"),
                                  loader.dataset, X, y_target, y_pred, idx)
        
            if save_output and self.test_output is not None:
                sid = "".join(loader.dataset.label_files[idx].split("/")[-1].split(".")[0:2])[3:]
                f = open(self.test_output, 'a')
                f.write(f'{sid} {loss_idx:>.4f}')
                for m in range(M):
                    f.write(f' {metrics_idx[m]:>.4f}')
                    f.write(f'\n')
                f.close()


                
    # Run at the end of each epoch
    def _epoch_end(self):
        # Print stuff
        if self.current_epoch % self.print_training_metrics_on_epoch == 0:
            print(f'Epoch={self.current_epoch} : loss={self.current_train_loss_avg:>.4f}, ' +
                  f'lr={self.scheduler.get_last_lr()[0]:>.5f}')

        
        # Save model
        if self.model_output is not None:
            torch.save({'epoch': self.current_epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'valid_loss': self.valid_loss[1]
                        }, self.model_output + "_last")
            if self.valid_loss[0] is not None:
                if self.valid_loss[1] < self.valid_loss[0]:
                    torch.save({'epoch': self.current_epoch,
                                'model_state': self.model.state_dict(),
                                'optimizer_state': self.optimizer.state_dict(),
                                'valid_loss': self.valid_loss[1]
                    }, self.model_output + "_best")

        # Update/reset
        self.scheduler.step()
        self.current_epoch += 1
        self.current_train_loss_avg = 0
        


        
    ### Function to write image data
    def _save_output(self, folder, dataset, inputs, target, output, idx):
        if folder is not None:
            basename = dataset.label_files[idx].split("/")[-1].split(".")[0:2]
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

            for i in range(len(dataset.image_files[idx])):
                input_str = dataset.image_files[idx][i].split(".")[-2:]
                input_path = os.path.join(folder, ".".join(basename) + "." + ".".join(input_str))
                dataset._save_output(inputs[:, i, ...], input_path, dtype=np.float32)
            target_path = os.path.join(folder, ".".join(basename) + ".target.mgz")
            dataset._save_output(target, target_path, dtype=np.int32, is_onehot=False)
            output_path = os.path.join(folder, ".".join(basename) + ".output.mgz")
            dataset._save_output(output, output_path, dtype=np.int32, is_onehot=False)
