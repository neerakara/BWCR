#!/bin/bash
#
# Script to send job to SLURM clusters using sbatch.
# Usage: sbatch main.sh
# Adjust line '-l hostname=xxxxx' before runing.
# The script also requires changing the paths of the CUDA and python environments and the code to the local equivalents of your machines.

## SLURM Variables:
#SBATCH  --output=/data/scratch/nkarani/logs/%j.out
#SBATCH  --partition=titan,2080ti,gpu
#SBATCH  --exclude=anise,curcum,sumac,fennel,urfa-biber,rosemary
#SBATCH  --gres=gpu:1
#SBATCH  --cpus-per-task=8
#SBATCH  --mem=12G
#SBATCH  --time=24:00:00
#SBATCH  --priority='TOP'

# activate virtual environment
source /data/vision/polina/users/nkarani/anaconda3/bin/activate env_crael

## EXECUTION OF PYTHON CODE:
python /data/vision/polina/users/nkarani/projects/crael/seg/train.py \
--cv_fold_num 1 \
--run_number 1 \
--data_aug_prob 0.5 \
--model_has_heads 1 \
--method_invariance 2 \
--lambda_consis 0.0 \
--alpha_layer 10.0

echo "Hostname was: `hostname`"
echo "Reached end of job file."