#!/bin/bash
anatomy='prostate'
dataset='RUNMC'
daprob=0.5
l0=0.0
l1=1.0
l1loss='ce'
l2=1.0
l2loss='ce'
runnum=2

for tem in 100.0
do
    for cvfold in 3
    do
        sbatch --no-requeue /data/vision/polina/users/nkarani/projects/crael/seg/train.sh $anatomy $dataset $cvfold $runnum $daprob $l0 $l1 $l2 $l1loss $l2loss $tem
    done
done