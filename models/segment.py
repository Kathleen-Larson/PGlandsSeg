import os
import time
import pathlib as Path
import numpy as np
import freesurfer as fs

import torch
import torch.optim
import torch.nn as nn
from torch.cuda.amp import autocast


class Segment:
    def __init__(self, model, optimizer, scheduler, loss, output_folder, start_full_aug_on, device):
        super().__init__()

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        
        self.start_full_aug_on = start_full_aug_on        
        self.output_folder = output_folder

        self.print_training_metrics_on_epoch = 1
        self.valid_loss = None
        
        if output_folder is not None:
            self.train_output = os.path.join(output_folder, "training_log.txt")
            self.valid_output = os.path.join(output_folder, "validation_log.txt")
            self.test_output = os.path.join(output_folder, "testing_log.txt")
            self.model_output = os.path.join(output_folder, "model")
        else:
            self.train_output = None
            self.valid_output = None
            self.test_output = None
            self.model_output = None
            
            
            
    ### Training loop
    def train(self, loader, metrics_list, epoch, save_output:bool=False):
        N = len(loader.dataset)
        M = len(metrics_list)
        loss_avg = 0
        metrics = np.zeros((M))
        
        if epoch < self.start_full_aug_on:
            augmentation = loader.dataset.base_augmentation
        else:
            augmentation = loader.dataset.full_augmentation
        
        self.model.zero_grad()
        self.model.train()
        
        for X, y, idx in loader:
            X, y = augmentation(X.to(self.device), y.to(self.device))
                        
            self.optimizer.zero_grad()
            with autocast():
                logits = self.model(X)
                loss = self.loss(logits, y)

            loss.backward()    
            loss_avg += loss.item()
            self.optimizer.step()
            
            y_target = torch.argmax(y, dim=1)
            y_pred = torch.argmax(logits, dim=1)
            
            if metrics_list is not None:
                metrics = [metrics[m] + metrics_list[m](y_pred, y_target) for m in range(M)]
            
            if save_output and self.output_folder is not None:
                self._save_output(os.path.join(self.output_folder, "train_data"),
                                  loader.dataset, X, y_target, y_pred, idx)
        loss_avg = loss_avg / N
        metrics = [metrics[m] / N for m in range(M)]
        
        
        if self.train_output is not None:
            f = open(self.train_output, 'a')
            f.write(f'{epoch} {loss_avg:>.4f}')
            for m in range(M):
                f.write(f' {metrics[m]:>.5f}')
            f.write(f'\n')
            f.close()
            
        
        if epoch % self.print_training_metrics_on_epoch == 0:
            print(f'Epoch={epoch} : loss={loss_avg:>.4f}, lr={self.scheduler.get_last_lr()[0]:>.5f}')
        

            
    ### Validation loop
    def validate(self, loader, metrics_list, epoch, save_output:bool=False):
        N = len(loader.dataset)
        M = len(metrics_list)
        loss_avg = 0
        metrics = np.zeros((M))
        
        if epoch < self.start_full_aug_on:
            augmentation = loader.dataset.base_augmentation
        else:
            augmentation = loader.dataset.full_augmentation
            
        self.model.eval()

        for X, y, idx in loader:
            X, y = augmentation(X.to(self.device), y.to(self.device))

            with torch.no_grad():
                logits = self.model(X)
                loss = self.loss(logits, y)
                loss_avg += loss.item()

            y_target = torch.argmax(y, dim=1)
            y_pred = torch.argmax(logits, dim=1)
            
            if metrics_list is not None:
                metrics = [metrics[m] + metrics_list[m](y_pred, y_target) for m in range(M)]

            if save_output and self.output_folder is not None:
                self._save_output(os.path.join(self.output_folder, "valid_data"),
                                  loader.dataset, X, y_target, y_pred, idx)
            
        loss_avg = loss_avg / N
        metrics = [metrics[m] / N for m in range(M)]
        self.scheduler.step()

        if self.valid_output is not None:
            f = open(self.valid_output, 'a')
            f.write(f'{epoch} {loss_avg:>.4f}')
            for m in range(M):
                f.write(f' {metrics[m]:>.5f}')
            f.write(f'\n')
            f.close()

        if self.model_output is not None:
            torch.save({'epoch': epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'valid_loss': loss_avg
                        }, self.model_output + "_last")
            if self.valid_loss is not None:
                if loss_avg < self.valid_loss:
                    torch.save({'epoch': epoch,
                                'model_state': self.model.state_dict(),
                                'optimizer_state': self.optimizer.state_dict(),
                                'valid_loss': loss_avg
                    }, self.model_output + "_best")
            else:
                self.valid_loss = loss_avg

        

            
    
    ### Testing loop
    def test(self, loader, metrics_list, save_output:bool=False):
        N = len(loader.dataset)
        M = len(metrics_list)

        loss_idx = 0.0
        metrics_idx = np.zeros((M, 1))
        augmentation = loader.dataset.base_augmentation

        self.model.eval()

        for X, y, idx in loader:
            X, y = augmentation(X.to(self.device), y.to(self.device))
            
            with torch.no_grad():
                logits = self.model(X)
                loss = self.loss(logits, y)
                loss_idx = loss.item()

            y_target = torch.argmax(y, dim=1)
            y_pred = torch.argmax(logits, dim=1)

            if metrics_list is not None:
                metrics_idx = [metrics_list[m](y_pred, y_target) for m in range(M)]

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
