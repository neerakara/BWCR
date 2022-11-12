#!/bin/bash
#
# Script to send job to SLURM clusters using sbatch.
# Usage: sbatch main.sh
# Adjust line '-l hostname=xxxxx' before runing.
# The script also requires changing the paths of the CUDA and python environments and the code to the local equivalents of your machines.

## SLURM Variables:
#SBATCH  --output=/data/scratch/nkarani/logs/%j.out
#SBATCH  --partition=gpu
#SBATCH  --exclude=anise,curcum,sumac,fennel,rosemary,urfa-biber,marjoram,mint
#SBATCH  --gres=gpu:1
#SBATCH  --cpus-per-task=8
#SBATCH  --mem=16G
#SBATCH  --time=02:00:00
#SBATCH  --priority='TOP'

# activate virtual environment
source /data/vision/polina/users/nkarani/anaconda3/bin/activate env_crael

## EXECUTION OF PYTHON CODE:
python /data/vision/polina/users/nkarani/projects/crael/seg/evaluate.py \
--cv_fold_num 2 \
--run_num 1 \
--test_sub_dataset 'BIDMC' \
--model_has_heads 1 \
--method_invariance 3 \
--lambda_data_aug 1.0 \
--lambda_consis 1.0 \
--alpha_layer 100.0

echo "Hostname was: `hostname`"
echo "Reached end of job file."