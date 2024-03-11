#!/bin/bash

#  SLURM STUFF
#SBATCH --account=lcnrtx
#SBATCH --partition=lcnrtx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --gpus=1
#SBATCH --time=0-80:00:00
#SBATCH --job-name=pseg_hypothalamus_setup

set -x

#run_type=slurm
run_type=debug
test_job_id=0

WDIR=/space/azura/1/users/kl021/Code/PituitarySegmentation
DATA_DIR=$WDIR/data/results/fsm

pretrain=$WDIR/pretrain.py
train=$WDIR/train.py

n_jobs=1
n_start=0
let n_jobs=$n_jobs-1

output_str='hypothalamus_seg_setup__mean_losses'


function call-train(){
    if [ $run_type == "slurm" ] ; then
        JOB_ID=$SLURM_ARRAY_TASK_ID
        n_workers=4
    else
        JOB_ID=$1
        n_workers=8
    fi
    
    # Define inputs
    aug_config=${WDIR}'/configs/augmentation/aug_config__hypothalamus_setup.txt'
    data_config=${WDIR}'/configs/data/data_config_t1.csv'
    batch_size=1
    start_aug_on=0
    steps_per_epoch=1000
    switch_loss_on=5
    max_n_epochs=205
    network='UNet3D_3layers'
    loss_fns=('mean_mse_loss_logits_yesbackground' 'mean_dice_loss_yesbackground')
    optim='Adam'
    lr_start=0.0001
    lr_param=0.95
    weight_decay=0.000001
    lr_scheduler='ConstantLR'
  
    metrics_train="MeanDice"
    metrics_valid="MeanDice"
    metrics_test="MeanDice"

    output_dir=${DATA_DIR}'/'${output_str}
    mkdir -p $output_dir

    model_state_path=${output_dir}"/model_best"

    # Run training
    python3 $train \
	    --aug_config $aug_config \
            --data_config $data_config \
            --loss_fns ${loss_fns[@]} \
            --lr_start $lr_start \
            --lr_param $lr_param \
            --lr_scheduler $lr_scheduler \
            --max_n_epochs $max_n_epochs \
            --n_workers $n_workers \
            --optim $optim \
            --output_dir $output_dir \
            --start_aug_on $start_aug_on \
            --steps_per_epoch $steps_per_epoch \
            --switch_loss_on $switch_loss_on \
            --weight_decay $weight_decay \
	    --model_state_path $model_state_path
}



function main(){
    if [ $run_type == 'slurm' ] ; then
	sbatch --output=slurm_outputs/${output_str}.out $0 call-train
	#sbatch --array=$n_start-$n_jobs --output=slurm_outputs/${output_str}_%a.out $0 call-train
    else
	call-train $test_job_id
    fi
}



if [[ $1 ]] ; then
    command=$1
    echo $1
    shift
    $command $@
else
    main
fi
