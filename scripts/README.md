# PGland segmentation

## Python Env. Management
If you're using Conda to maintain your python environment, use the following steps to setup the environment for this project:
1. conda create --name {name of your env} --file requirements.txt
2. conda actiavte {name of your env}


If you're using Poetry instead, use these steps:
1. poetry install
2. poetry shell
3. python run train.py


## Here is the general workflow and contents of each file:

Main scripts:
   - segment__*.sh : use to define input parameters and run through slurm as a job array
   - train.py : parses all input parameters (or default values), sets up and runs the training pipeline

W/in models/:
     - segment.py : contains the training, validation, and testing loops (data augmentation performed here)
     - loss_functions.py : contains all loss functions
     - metrics.py : contains several metrics used to quantify training progress (mean dice, surface distances)

W/in data_utils:
     - pituitarypineal.py : defines custom dataloader and sets up data augmentation
     - transforms.py : sets up data augmentation functions (interfaces with cornucopia repo)

W/in configs: these are files that define specific values for hyperparameter and network optimization:
     - optimizer :  optimizer type, lr_start, lr_scheduler, weight_decay, momentum
     - data : input modality combinations
     - model : number of layers, activation function, possibly more once I get there