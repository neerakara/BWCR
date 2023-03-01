#!/bin/bash
anatomy='acdc'
daprob=0.5
l0=0.0
l1=1.0
l2=1.0
tem=1.0
l1loss='ce'
l2loss='l2'
dist=1
runnum=1

for dataset in 'acdc'
do
    for cvfold in 1 2 3 10 20 30
    do
        sbatch --no-requeue /data/vision/polina/users/nkarani/projects/crael/seg/train.sh $anatomy $dataset $cvfold $runnum $daprob $l0 $l1 $l2 $l1loss $l2loss $dist $tem
    done
done