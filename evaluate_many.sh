#!/bin/bash
anatomy='prostate' # 'ms' / 'prostate' / acdc
dataset='nci' # nci / acdc
testdataset='nci' # nci / acdc
runnum=1
daprob=0.5
l0=0.0
l1=1.0
l2=1.0
l1loss='ce'
l2loss='l2'
dist=1
numlabels=3

for cvfold in 2
do
    sbatch /data/vision/polina/users/nkarani/projects/crael/seg/evaluate.sh $anatomy $dataset $testdataset $cvfold $runnum $daprob $l0 $l1 $l2 $l1loss $l2loss $dist $numlabels
done