import argparse
from datetime import datetime


def set_argparse_defs(parser):
    parser.set_defaults(accelerator='gpu')
    parser.set_defaults(devices=1)
    parser.set_defaults(num_sanity_val_steps=0)
    parser.set_defaults(deterministic=False)
    
    return parser



def add_argparse(parser):
    ### Data loader args
    parser.add_argument('--aug_config', default=None,
                        help='text file listing augmentation parameters')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='# samples in batch; default is 1')
    parser.add_argument('--input_data_files',
                        help='csv file listing filenames of input data')
    parser.add_argument('--data_inds_config', help='csv file listing indices '
                        'for training/validation/testing splits')
    parser.add_argument('--n_data_splits', type=int, default=3,
                        help='number of splits (1=train, 2=train+test, '
                        '3=train+valid+test; default is 3')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='number of workers for data loaders; default is '
                        '8')
    parser.add_argument('--output_dir', default=None,
                        help='output folder for model predictions')
    
    ## Data augmentation args
    parser.add_argument('--bias_max', type=float, default=1.,
                        help='maximum value of bias field in intensity '
                        'augmentation; default is 1')
    parser.add_argument('--bias_shape_factor', type=float, default=0.025,
                        help='ratio of small field to crop patch size during '
                        'intensity augmentation; default is 0.025')
    parser.add_argument('--bias_std', type=float, default=0.3,
                        help='stdev for bias field in intensity augmentation; '
                        'default is 0.3')
    parser.add_argument('--crop_patch_size', type=int, default=96,
                        help='size of cropped volume for spatial '
                        'augmentation; default is 96')
    parser.add_argument('--elastic_max_disp', type=float, default=0.1,
                        help='maximum displacement for spatial augmentation '
                        '(elastic); default is 0.1')
    parser.add_argument('--elastic_shape_factor', type=float, default=0.0625,
                        help='ratio of SVF size to image size for spatial '
                        'augmentation (elastic); default is 0.0625')
    parser.add_argument('--elastic_n_steps', type=int, default=7,
                        help='number of squaring/scaling steps during spatial '
                        'augmentation (elastic); default is 7')
    parser.add_argument('--gamma_std', type=float, default=0.5,
                        help='stdev of gamma function used in intensity '
                        'augmentation; default is 0.5')
    parser.add_argument('--noise_std', type=float, default=21.0,
                        help='stdev of gaussian white noise used in intensity '
                        'augmentation; default is 21.0')
    parser.add_argument('--norm_percent', type=float, default=0.95,
                        help='image intensity percentile used for robust '
                        'normalization; default is 0.95')
    parser.add_argument('--prob_lr_flip', type=float, default=0.5,
                        help='probability to flip across left-right axis '
                        'during data augmentation; default is 0.5')
    parser.add_argument('--rot_bounds', type=float, default=15.,
                        help='rotation bounds for spatial augmentation '
                        '(affine); default is 15')
    parser.add_argument('--scale_bounds', type=float, default=0.15,
                        help='scaling bounds for spatial augmentation '
                        '(affine); default is 0.15')
    parser.add_argument('--shear_bounds', type=float, default=0.012,
                        help='shearing bounds for spatial augmentation '
                        '(affine); default is 0.012')
    parser.add_argument('--trans_bounds', type=float, default=0.0,
                        help='translation bounds for spatial augmentation '
                        '(affine); default is 0')
    
    ### Model args
    parser.add_argument('--activation_function', default='ELU',
                        help='activation function for UNet')
    parser.add_argument('--conv_window_size', type=int, default=3,
                        help='size of convolution kernel (isotropic)')
    parser.add_argument('--max_n_epochs', type=int,
                        help='number of training epochs')
    parser.add_argument('--max_n_steps', type=int,
                        help='number of training steps')
    parser.add_argument('--model_state_path', default=None,
                        help='path to model state to load, if any (required '
                        'for predict.py)')
    parser.add_argument('--use_residuals', type=int, default=0,
                        help='use residual connections within model')
    parser.add_argument('--n_unet_levels', type=int, default=4,
                        help='number of levels for unet configuration')
    parser.add_argument('--pooling_function', default='MaxPool',
                        help='pooling function for UNet (must be part of '
                        'torch.nn)')
    parser.add_argument('--pool_window_size', type=int, default=2,
                        help='size of pooling window (isotropic)')
    parser.add_argument('--pretrain_model_path', default=None,
                        help='path to model used for pretraining')
    parser.add_argument('--steps_per_epoch', type=int,
                        help='number of steps per epoch')

    ### Optimizer args
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='lower beta value for optimizer (Adam/AdamW); '
                        'default is 0.9')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='higher beta value for optimizer (Adam/AdamW); '
                        'default is 0.99')
    parser.add_argument('--dampening', type=float, default=0,
                        help='dampening for optimizer (SGD); default is 0')
    parser.add_argument('--loss_fns', nargs='+',
                        default='mean_dice_loss_yesbackground',
                        help='training loss function (from within '
                        'models/loss_functions.py); default is '
                        'mean_dice_loss_yesbackground')
    parser.add_argument('--lr_start', type=float, default=0.0001,
                        help='initial learning rate; default is 1e-4')
    parser.add_argument('--momentum', type=float, default=0.,
                        help='momentum for optimizer (SGD); default is 0')
    parser.add_argument('--optimizer', default='Adam',
                        help='optimizer for training (case-sensitive); '
                        'default is Adam')
    parser.add_argument('--switch_loss_on', type=int,
                        help='number of steps after which to switch losses '
                        '(if using 2)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='training weight decay; default is 0')
    
    ### Other
    parser.add_argument('--mode', type=str, default='predict',
                        help='Specify mode (train/predict); default is '
                        'predict')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed; default is 0')
    parser.add_argument('--save_output_every', type=int, default=0,
                        help='save model state and output data every N '
                        'epochs; default is 0=off')
    
    return parser
