#!/bin/bash
#
# Script to send job to SLURM clusters using sbatch.
# Usage: sbatch main.sh
# Adjust line '-l hostname=xxxxx' before runing.
# The script also requires changing the paths of the CUDA and python environments and the code to the local equivalents of your machines.

## SLURM Variables:
#SBATCH  --output=/data/scratch/nkarani/logs/%j.out
#SBATCH  --partition=titan,2080ti,gpu
#SBATCH  --exclude=anise,curcum,sumac,fennel,urfa-biber,rosemary,mace,malt
#SBATCH  --gres=gpu:1
#SBATCH  --cpus-per-task=8
#SBATCH  --mem=12G
#SBATCH  --time=02:00:00
#SBATCH  --priority='TOP'

# activate virtual environment
source /data/vision/polina/users/nkarani/anaconda3/bin/activate env_crael

## EXECUTION OF PYTHON CODE:
python /data/vision/polina/users/nkarani/projects/crael/seg/evaluate.py \
--dataset $1 \
--sub_dataset $2 \
--test_sub_dataset $3 \
--cv_fold_num $4 \
--run_number $5 \
--data_aug_prob $6 \
--l0 $7 \
--l1 $8 \
--l2 $9 \
--l1_loss ${10} \
--l2_loss ${11} \
--weigh_lambda_con ${12} \
--num_labels ${13} \
--out_layer_type ${14}

echo "Hostname was: `hostname`"
echo "Reached end of job file."