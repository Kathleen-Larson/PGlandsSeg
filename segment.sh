#!/bin/bash

#  SLURM STUFF
#SBATCH --account=lcnrtx
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --gpus=1
#SBATCH --time=0-04:00:00
#SBATCH --job-name=pseg2

WDIR=/space/azura/1/users/kl021/Code/PituitarySegmentation
n_jobs=$(ls $WDIR/configs/optimizer/optimizer_config_*.txt | wc -l)

#run_type=slurm
run_type=debug
test_job_id=0

n_workers=4
n_layers=3

n_start=0
let n_jobs=$n_jobs-1

loss_fns='mean_mse_loss_logits_yesbackground'
output_str=${loss_fns}'_'${n_layers}'layers'


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
    max_n_epochs=300
    network='UNet3D_'${n_layers}'layers'
    optim='Adam'
    #loss_fns='mean_mse_loss_logits'
    lr_start=0.0001
    lr_param=0.95
    weight_decay=0.000
    lr_scheduler='ConstantLR'
  
    metrics_train="MeanDice"
    metrics_valid="MeanDice"
    metrics_test="MeanDice"
    
    load_model_state=last
    output_dir=${WDIR}'/data/results/fsm/'${output_str}
    mkdir -p $output_dir
    
    # Run train
    python3 train.py \
	    --data_config $data_config \
	    --weight_decay $weight_decay \
	    --loss_fns $loss_fns \
	    --lr_start $lr_start \
	    --lr_param $lr_param \
            --lr_scheduler $lr_scheduler \
	    --max_n_epochs $max_n_epochs \
	    --n_workers $n_workers \
	    --optim $optim \
	    --output_dir $output_dir \
	    --load_model_state $load_model_state
}



function main(){
    if [ $run_type == 'slurm' ] ; then
	sbatch --array=0 --output=slurm_outputs/${output_str}.out $0 call-train
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
