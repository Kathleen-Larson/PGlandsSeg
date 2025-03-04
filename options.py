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
    parser.add_argument('--aug_config', dest='aug_config', default=None, \
                        help='text file listing augmentation parameters')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, \
                        help='# samples in batch')
    parser.add_argument('--input_data_files', dest='input_data_files', \
                        help='csv file listing filenames of input data')
    parser.add_argument('--data_inds_config', dest='data_inds_config', \
                        help='csv file listing indices for training/validation/testing splits')
    parser.add_argument('--n_data_splits', dest='n_data_splits', type=int, default=3, \
                        help='number of splits (1=train, 2=train+test, 3=train+valid+test')
    parser.add_argument('--n_workers', dest='n_workers', type=int, default=8, \
                        help='number of workers for data loaders')
    parser.add_argument('--output_dir', dest='output_dir', default=None, \
                        help='output folder for model predictions')
    parser.add_argument('--output_fname_base', dest='output_fname_base', default=None, \
                        help='file basename for output (use when processing single image)')
    
    ## Data augmentation args
    parser.add_argument('--crop_patch_size', dest='crop_patch_size', type=int, default=None, \
                        help='size of cropped volume for data augmentation')
    parser.add_argument('--use_crop_template', dest='use_crop_template', type=int, default=0, \
                        help='use template to determine crop window')
    parser.add_argument('--noise_std_additional', dest='noise_std_additional', type=float, default=0.0, \
                        help='std dev of gaussian white noise added to input data (for noise sensitivity testing)')
    parser.add_argument('--start_aug_on', dest='start_aug_on', type=int, default=None, \
                        help='epoch on which to start full data augmentation')
    parser.add_argument('--segment_pituitary', dest='segment_pituitary', type=int, default=1, \
                        help='flag to turn on/off segmentation of pituitary labels')
    parser.add_argument('--segment_pineal', dest='segment_pineal', type=int, default=1, \
                        help='flag to turn on/off segmentation of pineal label')
    parser.add_argument('--apply_postpit_aug', dest='apply_postpit_aug', type=int, default=0, \
                        help='flag to turn on/off histogram matching between ant. pit. and post. pit. labels')
    parser.add_argument('--apply_pineal_aug', dest='apply_pineal_aug', type=int, default=0, \
                        help='flag to turn on/off simulation of pineal calcificiation')
    parser.add_argument('--apply_robust_normalization', dest='apply_robust_normalization', type=int, default=1, \
                        help='flag to turn on/off robust normalization')
    
    
    ### Metrics args
    parser.add_argument('--metrics_train', dest='metrics_train', nargs='+', default=[], \
                        help='training metrics')
    parser.add_argument('--metrics_test', dest='metrics_test', nargs='+', default=[], \
                        help='testing metrics')
    parser.add_argument('--metrics_valid', dest='metrics_valid', nargs='+', default=[], \
                        help='validation metrics')

    ### Model args
    parser.add_argument('--activation_function', dest='activation_function', default='ELU', \
                        help='activation function for UNet')
    parser.add_argument('--conv_window_size', dest='conv_window_size', type=int, default=3, \
                        help='size of convolution kernel (isotropic)')
    parser.add_argument('--n_levels', dest='n_levels', type=int, default=3, \
                        help='number of levels for UNet')
    parser.add_argument('--max_n_epochs', dest='max_n_epochs', type=int, \
                        help='number of training epochs')
    parser.add_argument('--max_n_steps', dest='max_n_steps', type=int, \
                        help='number of training steps')
    parser.add_argument('--model_state_path', dest='model_state_path', default=None, \
                        help='path to model state to load, if any (required for predict.py)')
    parser.add_argument('--use_residuals', dest='use_residuals', type=int, default=0, \
                        help='use residual connections within model')
    parser.add_argument('--network', dest='network', default='UNet3D_4levels', \
                        help='training network')
    parser.add_argument('--pooling_function', dest='pooling_function', default='MaxPool', \
                        help='pooling function for UNet')
    parser.add_argument('--pool_window_size', dest='pool_window_size', type=int, default=2, \
                        help='size of pooling window (isotropic)')
    parser.add_argument('--pretrain_model_path', dest='pretrain_model_path', default=None, \
                        help='path to model used for pretraining')
    parser.add_argument('--steps_per_epoch', dest='steps_per_epoch', type=int, \
                        help='number of steps per epoch')

    ### Optimizer args
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, \
                        help='lower beta value for optimizer (Adam/AdamW)')
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.99, \
                        help='higher beta value for optimizer (Adam/AdamW)')
    parser.add_argument('--dampening', dest='dampening', type=float, default=0, \
                        help='dampening for optimizer (SGD)')
    parser.add_argument('--loss_fns', dest='loss_fns', nargs='+', default='dice_cce_loss', \
                        help='training loss function')
    parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=0, \
                        help='learning rate decay')
    parser.add_argument('--lr_param', dest='lr_param', type=float, default=0.95, \
			help='parameter for lr_scheduler (function dependent')
    parser.add_argument('--lr_scheduler', dest='lr_scheduler', default='ConstantLR', \
                        help='lr_scheduler for optimizer (must be part of torch.optim')
    parser.add_argument('--lr_start', dest='lr_start', type=float, default=0.001, \
                        help='initial learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default='0', \
                        help='momentum for optimizer (SGD)')
    parser.add_argument('--optimizer', dest='optimizer', default='Adam', \
                        help='optimizer for training (case-sensitive)')
    parser.add_argument('--switch_loss_on', dest='switch_loss_on', type=int, \
                        help='number of steps after which to switch from MSE to Dice loss')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.0, \
                        help='training weight decay')
    
    ### Other
    parser.add_argument('--seed', dest='seed', type=int, default=0, \
                        help='random seed for xval')
    parser.add_argument('--save_output_every', dest='save_output_every', type=int, default=0, \
                        help='save model state and output data every N epochs')

    
    return parser
