#!/bin/bash
#
# Script to send job to SLURM clusters using sbatch.
# Usage: sbatch train.sh
# Adjust line '-l hostname=xxxxx' before runing.
# The script also requires changing the paths of the CUDA and python environments and the code to the local equivalents of your machines.

## SLURM Variables:
#SBATCH  --output=/data/scratch/nkarani/logs/%j.out
#SBATCH  --partition=titan,2080ti,gpu
#SBATCH  --exclude=anise,curcum,sumac,fennel,urfa-biber,rosemary,mace,malt
#SBATCH  --gres=gpu:1
#SBATCH  --cpus-per-task=8
#SBATCH  --mem=12G
#SBATCH  --time=24:00:00
#SBATCH  --priority='TOP'

# activate virtual environment
source /data/vision/polina/users/nkarani/anaconda3/bin/activate env_crael

## EXECUTION OF PYTHON CODE:
python /data/vision/polina/users/nkarani/projects/crael/seg/train2.py \
--dataset $1 \
--sub_dataset $2 \
--cv_fold_num $3 \
--run_number $4 \
--data_aug_prob $5 \
--l0 $6 \
--l1 $7 \
--l2 $8 \
--l1_loss $9 \
--l2_loss ${10} \
--alpha_layer ${11} \
--temp ${12}

echo "Hostname was: `hostname`"
echo "Reached end of job file."