#!/bin/bash
anatomy='prostate'
daprob=0.5
l0=0.0
l1=1.0
l2=1.0
tem=1.0
l1loss='ce'
l2loss='l2'
runnum=1

for dataset in 'RUNMC'
do
    for cvfold in 3
    do
        sbatch --no-requeue /data/vision/polina/users/nkarani/projects/crael/seg/train.sh $anatomy $dataset $cvfold $runnum $daprob $l0 $l1 $l2 $l1loss $l2loss $tem
    done
done