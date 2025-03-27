# PGland segmentation

## Python Env. Management
If you're using Conda to maintain your python environment, use the following steps to setup the environment for this project:
1. conda create --name {name of your env} --file requirements.txt
2. conda actiavte {name of your env}

## General workflow:
- Run pglands_seg/train.py for training and pglands_seg/predict.py for inference
- All tuneable input parameters are configured in configs/train.yaml or configs/predict.yaml
- User can specify the following commandline arguments:
  1. --config - specifies the .yaml file containing all input parameters. If none is specified, the default is configs/train.yaml.
  2. --use_cuda - utilize CUDA for gpu support (if available, will default to cpu only if not specified)
  3. --resume - resumes training from a previously saved checkpoint (requires the user to specify the model checkpoint path in the config file)