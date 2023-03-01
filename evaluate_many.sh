#!/bin/bash
anatomy='acdc' # 'ms' / 'prostate'
dataset='acdc'
runnum=1
daprob=0.5
l0=0.0
l1=1.0
l2=0.1
l1loss='ce'
l2loss='l2'

for cvfold in 1 2 3 10
do
    sbatch /data/vision/polina/users/nkarani/projects/crael/seg/evaluate.sh $anatomy $dataset $cvfold $runnum $daprob $l0 $l1 $l2 $l1loss $l2loss
done