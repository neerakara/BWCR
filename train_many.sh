#!/bin/bash
anatomy='acdc'
daprob=0.5
l0=0.0
l1=1.0
l2=0.1
tem=1.0
l1loss='dice'
l2loss='l2'
alpha=0.1
runnum=3

for dataset in 'acdc'
do
    for cvfold in 1
    do
        sbatch --no-requeue /data/vision/polina/users/nkarani/projects/crael/seg/train.sh $anatomy $dataset $cvfold $runnum $daprob $l0 $l1 $l2 $l1loss $l2loss $alpha $tem
    done
done