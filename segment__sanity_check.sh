#!/bin/bash

#  SLURM STUFF
#SBATCH --account=lcnrtx
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --time=1-04:00:00
#SBATCH --job-name=pseg2

WDIR=/space/azura/1/users/kl021/Code/PituitarySegmentation
n_jobs=$(ls $WDIR/configs/optimizer/optimizer_config_*.txt | wc -l)

set -x

run_type=slurm
#run_type=debug
test_job_id=0

n_workers=4
n_layers=3

n_start=0
let n_jobs=$n_jobs-1


function call-train(){
    if [ $run_type == "slurm" ] ; then
        JOB_ID=$SLURM_ARRAY_TASK_ID
    else
	n_workers=8
        JOB_ID=$1
    fi
    
    # Define inputs
    data_config='configs/data/data_config_t1.csv'
    batch_size=1
    max_n_epochs=10
    network='UNet3D_3layers'
    optim='Adam'
    loss='dice_cce_loss'
    lr_start=0.001
    weight_decay=0.002
    lr_scheduler='ConstantLR'
  
    metrics_train="MeanDice"
    metrics_valid="MeanDice"
    metrics_test="MeanDice"

    output_dir=$WDIR/data/results/sanity_check
    mkdir -p $output_dir
    
    # Run train
    python3 train.py \
	    --data_config $data_config \
	    --weight_decay $weight_decay \
	    --lr_start $lr_start \
	    --max_n_epochs $max_n_epochs \
	    --metrics_test $metrics_test \
	    --metrics_train $metrics_train \
	    --metrics_valid $metrics_valid \
	    --n_workers $n_workers \
	    --optim $optim \
	    --lr_scheduler $lr_scheduler \
	    --output_dir $output_dir
}



function main(){
    if [ $run_type == 'slurm' ] ; then
	sbatch --array=0 --output=slurm_outputs/sanity_check.out $0 call-train
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
