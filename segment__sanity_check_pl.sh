#!/bin/bash

#  SLURM STUFF
#SBATCH --account=lcnrtx
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH --time=1-08:00:00
#SBATCH --job-name=ppseg_pl

csv_t1=dataset/data_config_t1.csv
csv_t2=dataset/data_config_t2.csv
csv_qt1=dataset/data_config_qt1.csv
csv_pd=dataset/data_config_pd.csv

csv_t1t2=dataset/data_config_t1t2.csv
csv_t1qt1=dataset/data_config_t1qt1.csv
csv_t1pd=dataset/data_config_t1pd.csv

csv_t1t2pd=dataset/data_config_t1t2pd.csv
csv_t1t2qt1=dataset/data_config_t1t2qt1.csv
csv_t1pdqt1=dataset/data_config_t1pdqt1.csv
csv_t2pdqt1=dataset/data_config_t2pdqt1.csv
csv_all=dataset/data_config_all.csv

set -x

run_type=slurm
#run_type=debug

n_workers=4
n_layers=3

input_data_list=('t1')
n_jobs=${#input_data_list[@]}
let n_jobs=$n_jobs-1

function call-train_crop(){
    if [ $run_type == "slurm" ] ; then
        JOB_ID=$SLURM_ARRAY_TASK_ID
    else
	n_workers=8
        JOB_ID=$1
    fi
    input_data=${input_data_list[$JOB_ID]}
    
    # Define inputs
    seed=0
    data_config='configs/data/data_config_'$input_data'.csv'
    batch_size=1
    max_n_epochs=10 #300
    network='UNet3D_'${n_layers}'layers'
    optim='Adam'
    loss='dice_cce_loss'
    lr_start=0.001
    weight_decay=0.002
    lr_scheduler='ConstantLR'

    metrics_train="MeanDice" # HausDist2"
    metrics_valid="MeanDice" # HausDist2"
    metrics_test="MeanDice" # HausDist2"
    output_dir='data/results/sanity_check_pl'
    mkdir -p $output_dir
    
    # Run train
    python3 train_pl.py \
	    --data_config $data_config \
	    --weight_decay $weight_decay \
            --lr_start $lr_start \
            --max_n_epochs $max_n_epochs \
	    --metrics_test $metrics_test \
	    --metrics_train $metrics_train \
	    --metrics_valid $metrics_valid \
	    --n_workers $n_workers \
	    --optim $optim \
	    --output_dir $output_dir \
	    --lr_scheduler $lr_scheduler 
}



function main(){
    if [ $run_type == 'slurm' ] ; then
	sbatch --array=0 --output=slurm_outputs/sanity_check_pl.out $0 call-train_crop
    else
	call-train_crop 0
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
