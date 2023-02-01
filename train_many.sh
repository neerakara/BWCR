#!/bin/bash
anatomy='prostate'
dataset='BMC'
daprob=0.5
l0=1.0
l1=0.0
l1loss='ce'
tem=1.0
l2loss='l2'
runnum=1

for l2 in 0.0
do
    for cvfold in 1 10 3 30
    do
        sbatch --no-requeue /data/vision/polina/users/nkarani/projects/crael/seg/train.sh $anatomy $dataset $cvfold $runnum $daprob $l0 $l1 $l2 $l1loss $l2loss $tem
    done
done