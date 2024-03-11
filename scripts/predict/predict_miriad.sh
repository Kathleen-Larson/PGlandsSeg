#!/bin/bash

#  SLURM STUFF
#SBATCH --account=lcnrtx
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --gpus=1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=pseg_dice

set -x

WDIR=/space/azura/1/users/kl021/Code/PituitarySegmentation
DATA_DIR=$WDIR/data/results
DATASET=miriad
#n_jobs=$(ls $WDIR/configs/optimizer/optimizer_config_*.txt | wc -l)

#run_type=slurm
run_type=debug
test_job_id=0

n_workers=4
n_layers=3

n_jobs=${#loss_fn_list[@]}
n_start=0
let n_jobs=$n_jobs-1

output_str='testing'


function call-train(){
    if [ $run_type == "slurm" ] ; then
        JOB_ID=$SLURM_ARRAY_TASK_ID
    else
	n_workers=8
        JOB_ID=$1
    fi
    
    # Define inputs
    data_config='configs/data/data_config__miriad.csv'
    batch_size=1
    network='UNet3D_'${n_layers}'layers'
    metrics_test='MeanDice'
    
    train_loss='mean_dice_loss_yesbackground'
    model_path=${DATA_DIR}'/fsm/augmentation_noise_testing__sigma000/model_best'
    output_dir=${DATA_DIR}'/'${DATASET}'/testing'
    mkdir -p $output_dir
    
    # Run train
    python3 predict.py \
	    --data_config $data_config \
	    --n_workers $n_workers \
	    --output_dir $output_dir \
	    --model_state_path $model_path
}



function main(){
    if [ $run_type == 'slurm' ] ; then
	sbatch --array=$n_start-$n_jobs --output=slurm_outputs/${output_str}_%a.out $0 call-train
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
