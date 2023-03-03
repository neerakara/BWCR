#!/bin/bash
anatomy='acdc' # prostate / acdc
daprob=0.5
l0=0.0
l1=1.0
l2=1.0
numlabels=4
l1loss='ce'
l2loss='l2'
dist=1
runnum=2
outlayer=1

for dataset in 'acdc'
do
    for cvfold in 1
    do
        sbatch --no-requeue /data/vision/polina/users/nkarani/projects/crael/seg/train.sh $anatomy $dataset $cvfold $runnum $daprob $l0 $l1 $l2 $l1loss $l2loss $dist $numlabels $outlayer
    done
done