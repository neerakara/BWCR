#!/bin/bash
anatomy='prostate' # 'ms' / 'prostate' / acdc
runnum=1
daprob=0.5
l0=0.0
l1=1.0
l2=1.0
l1loss='ce'
l2loss='l2'
dist=1
outlayer=1

if [ "$anatomy" = "acdc" ]; then
    numlabels=4
    dataset="acdc"
    testdataset="acdc"
elif [ "$anatomy" = "prostate" ]; then
    numlabels=3
    dataset="nci"
    testdataset="nci"
else
    echo "I do not recognize this anatomy."
fi

for cvfold in 100
do
    sbatch /data/vision/polina/users/nkarani/projects/crael/seg/evaluate.sh $anatomy $dataset $testdataset $cvfold $runnum $daprob $l0 $l1 $l2 $l1loss $l2loss $dist $numlabels $outlayer
done