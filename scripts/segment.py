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





class PGlandsSegmenter:
    def __init__(self, model, optimizer, device, **config):
        super().__init__()
        
        # Parse non-config args
        self.device = device
        self.model = model
        self.optimizer = optimizer
        
        # Parse config args
        self.checkpoint_path = config['checkpoint_path'] \
            if 'checkpoint_path' in config else None
        self.loss_funcs = config['loss_funcs']
        self.max_n_epochs = config['max_n_epochs'] \
            if 'max_n_epochs' in config else None
        self.max_n_steps = config['max_n_steps'] \
            if 'max_n_steps' in config else None
        self.output_dir = config['output_dir'] \
            if 'output_dir' in config else None
        self.save_outputs_every = config['save_outputs_every'] \
            if 'save_outputs_every' in config else None
        self.start_aug_on = config['start_aug_on'] \
            if 'start_aug_on' in config else None
        self.steps_per_epoch = config['steps_per_epoch'] \
            if 'steps_per_epoch' in config else None
        self.switch_loss_on = config['switch_loss_on'] \
            if 'switch_loss_on' in config else None

        # Initialize loss storage
        self.current_step = 0
        self.current_train_loss = [None, None]
        self.current_valid_loss = [None, None]
        self.start_epoch = 0
        
        # Load state (if necessary)
        if self.checkpoint_path is not None:
            if not os.path.isfile(self.checkpoint_path):
                utils.fatal(f'Training checkpoint {self.checkpoint_path} not '
                            + 'found')
            checkpoint = torch.load(self.checkpoint_path)
            """
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.start_epoch = checkpoint['epoch']
            self.current_train_loss = ['minimum', 'last']
            self.current_valid_loss = ['minimum', 'last']
            """

        # Set up output logs
        if self.output_dir is not None:
            self.model_output = os.path.join(output_folder, 'model')

            self.train_output = os.path.join(output_folder, 'train_log.txt')
            utils.
            self.valid_output = os.path.join(output_folder, 'valid_log.txt')
            self.test_output = os.path.join(output_folder, 'test_log.txt')
            self.model_output = os.path.join(output_folder, 'model')
        else:
            self.train_output = None
            self.valid_output = None
            self.test_output = None
            self.model_output = None


           if output_folder is not None and train_checkpoint_path is None:
        if not os.path.exists(output_folder):  os.mkdir(output_folder)

        _write_data_ids_log(os.path.join(output_folder, "train_data_ids_log.txt\
"), train_data)
        _setup_log(os.path.join(output_folder, "train_log.txt"), "Epoch  TrainL\
oss")

        if valid_data is not None:
            _write_data_ids_log(os.path.join(output_folder, "valid_data_ids_log\
.txt"), valid_data)
            _setup_log(os.path.join(output_folder, "valid_log.txt"), "Epoch  Va\
lidLoss")

        if test_data is not None:
            _write_data_ids_log(os.path.join(output_folder, "test_data_ids_log.\
txt"), test_data)
    

        
    ### Training loop
    def train(self, loader):
        N = len(loader.dataset)
        loss_avg = 0.0

        self.model.train()
        self.model.zero_grad()
        
        for data, idx in loader:
            X, y, template = loader.dataset.full_augmentations([x.to(self.device) for x in data])
            logits = self.model(X)

            loss = self.loss_fn(logits, y, gpu=True)
            loss_avg += loss.item()
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.current_step += 1
            self.current_train_loss_avg = loss_avg / self.current_step

            if self.steps_per_epoch > 10:
                if self.current_step % (self.steps_per_epoch // 10) == 0:
                    print(f'Epoch={self.current_epoch}, step={self.current_step}: loss={loss.item():>.4f}')
            else:
                print(f'Epoch={self.current_epoch}, step={self.current_step}: loss={loss.item():>.4f}')

        self.current_train_loss_avg = loss_avg / self.steps_per_epoch
        if self.train_output is not None:
            f = open(self.train_output, 'a')
            f.write(f'{self.current_epoch:5} {self.current_train_loss_avg:>9.4f}')
            f.write(f'\n')
            f.close()            


            
    ### Validation loop
    def validate(self,
                 loader,
                 save_dir:str=None,
                 save_output:bool=False,
                 write_inputs=True,
                 write_targets=False
    ):
        N = len(loader.dataset)
        loss_avg = 0.
        
        self.model.eval()
        self.model.zero_grad()
        
        for data, idx in loader:
            X, y = loader.dataset.base_augmentations(data)
            with torch.no_grad():
                logits = self.model(X)
            loss = self.loss_fn(logits, y, gpu=True)

            loss_avg += loss.item()
            y_target = torch.argmax(y, dim=1)
            y_pred = torch.argmax(F.softmax(logits, dim=1), dim=1) 
            
            if save_output and self.output_dir is not None:
                output_folder_valid = os.path.join(self.output_dir, "valid_data",
                                                   "epoch_" + str(self.current_epoch + 1).zfill(len(str(self.max_n_epochs)))) \
                                                   if save_dir is None else save_dir

                self._save_output(output_folder_valid, loader.dataset, X, y_target, y_pred, idx,
                                  write_targets=write_targets, write_inputs=write_inputs)
        loss_avg = loss_avg / N

        if self.valid_output is not None:
            f = open(self.valid_output, 'a')
            f.write(f'{self.current_epoch} {loss_avg:>9.4f}')
            f.write(f'\n')
            f.close()
        

    ### Testing loop
    def test(self,
             loader,
             save_output:bool=False,
             save_dir:str=None,
             mode:str='train',
             write_inputs:bool=True,
             write_targets:bool=False,
             write_posteriors:bool=True,
             output_basename=None,
    ):
        save_dir = self.output_dir if save_dir is None else os.path.join(self.output_dir, save_dir)
        
        N = len(loader.dataset)
        self.model.eval()

        for data, idx in loader:
            if mode == 'predict':
                X = loader.dataset.base_augmentations([x.to(self.device) for x in data])[0]
                X = X[0] if isinstance(X, list) else X
                y = None
            else:
                X, y = loader.dataset.base_augmentations([x.to(self.device) for x in data])[0:-1]

            with torch.no_grad():
                logits = self.model(X)
            posteriors = F.softmax(logits, dim=1)

            if save_output:
                self._save_output(save_dir, loader.dataset,
                                  X, y, posteriors, idx,
                                  output_basename=output_basename,
                                  write_inputs=write_inputs,
                                  write_targets=write_targets,
                                  write_posteriors=write_posteriors)

                
    # Run at the end of each epoch
    def _epoch_end(self, output_model:bool=False):
        # Save model
        save_model_every_epoch = 20
        save_model = (self.current_epoch + 1) % 20 == 0
        if self.model_output is not None and save_model:
            torch.save({'epoch': self.current_epoch + 1,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
            }, self.model_output + "_last.tar")        
        
        # Update/reset
        self.current_step = 0
        self.current_epoch += 1
        self.current_train_loss_avg = 0

        if self.current_epoch == self.switch_loss_on:
            self.loss_fn = self.loss_functions[1]
            print('Switching from', self.loss_functions[0].__name__, 'to', self.loss_functions[1].__name__, \
                  'after ', self.current_epoch, ' epochs')


        
    ### Function to write image data
    def _save_output(self, folder, dataset, inputs, target, output, idx,
                     output_basename=None, crop_type='bbox', conform2orig=True,
                     write_inputs=False, write_targets=False, write_posteriors=False):
        
        if folder is not None:
            if output_basename is not None:
                basename = [output_basename] if not isinstance(output_basename, list) \
                    else output_basename

            elif target is not None:
                basename = dataset.label_files[idx].split("/")[-1].split(".")[0]
                basename = [basename] if not isinstance(basename, list) else basename

            else:
                basename = dataset.image_files[idx][0].split("/")[-1].split(".")[0].split("_")[:-1]
                basename = [basename] if not isinstance(basename, list) else basename
                
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

            if write_posteriors:
                posterior = sf.Volume(np.squeeze(output.cpu().numpy()))
                posterior_path = os.path.join(folder, "_".join(basename + ["posterior.mgz"]))
                posterior.save(posterior_path)


            output_segmentation = torch.argmax(output, dim=1)
            output_path = os.path.join(folder, "_".join(basename + ["prediction.mgz"]))
            dataset._save_output(output_segmentation,
                                 output_path, idx, geom_idx=0, dtype=np.int32, crop_type=crop_type,
                                 conform2orig=conform2orig, is_labels=True, convert_labels=True)                
                
            if write_targets and target is not None:
                target_segmentation = torch.argmax(target, dim=1)
                target_path = os.path.join(folder, "_".join(basename + ["target.mgz"]))
                dataset._save_output(target_segmentation,
                                     target_path, idx, geom_idx=0, dtype=np.int32, crop_type=crop_type,
                                     conform2orig=conform2orig, is_labels=True, convert_labels=True)
            
            if write_inputs:
                for i in range(len(dataset.image_files[idx])):
                    input_str = dataset.image_files[idx][i].split("/")[-1].split(".")[0]
                    input_path = os.path.join(folder, "_".join([input_str, "input.mgz"]))                    
                    dataset._save_output(inputs[:, i, ...], input_path, idx, geom_idx=i, \
                                         dtype=np.float32, conform2orig=conform2orig)

