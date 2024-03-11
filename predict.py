import os, warnings, argparse, logging, random
from os import system
import numpy as np
import nibabel as nib
import freesurfer as fs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader

from data_utils.pituitarypineal_predict import call_dataset
from data_utils import transforms_singleinput as t

import options
import models
from models import unet, metrics



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
    
    ### Data set-up ###
    batch_size = pargs.batch_size
    data_config = pargs.data_config
    n_workers = pargs.n_workers

    data = call_dataset(data_config=data_config, crop_patch_size=(160, 160, 160))
    DL = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    
    ### Model set-up ###
    model = models.unet.__dict__[pargs.network](in_channels=data.__numinput__(),
                                                  out_channels=data.__numclass__(),
    ).to(device)

    model_state_path = pargs.model_state_path
    if model_state_path is not None:
        trained_model = torch.load(model_state_path)
        model.load_state_dict(trained_model["model_state"])
    else:
        raise Exception('model_state_path must be provided')

    
    ## Predict label maps for input dataset
    output_dir = pargs.output_dir

    model.eval()
    for X, idx in DL:
        # Do the predictions
        X = DL.dataset.augmentation(X.cuda())
        while len(X.shape) < 5: # idk WHY this is necessary and it's so annoying
            X = torch.unsqueeze(X, dim=0)

        with torch.no_grad():
            logits = model(X)
        y = torch.argmax(F.softmax(logits, dim=1), dim=1)
        breakpoint()
        
        # Write out the data
        basename = DL.dataset.image_files[idx][0].split("/")[-1].split(".")[0]
        for i in range(len(DL.dataset.image_files[idx])):
            input_path = os.path.join(output_dir, basename + "_input.mgz")
            DL.dataset._save_output(X[:, i, ...], input_path, dtype=np.float32)

        output_path = os.path.join(output_dir, basename + "_prediction.mgz")
        DL.dataset._save_output(y, output_path, dtype=np.int32)
        breakpoint()


    
    

##########################
if __name__ == "__main__":
    main(_config())
