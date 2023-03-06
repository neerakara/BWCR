#!/bin/bash
anatomy="acdc" # prostate / acdc
daprob=0.5
l0=0.0
l1=1.0
l1loss="ce"
l2loss="l2"
runnum=1
outlayer=1
dist=1

if [ "$anatomy" = "acdc" ]; then
    numlabels=4
    dataset="acdc"
elif [ "$anatomy" = "prostate" ]; then
    numlabels=3
    dataset="nci"
else
    echo "I do not recognize this anatomy."
fi

for cvfold in 200 300
do
    for l2 in 1.0
    do
        sbatch --no-requeue /data/vision/polina/users/nkarani/projects/crael/seg/train.sh $anatomy $dataset $cvfold $runnum $daprob $l0 $l1 $l2 $l1loss $l2loss $dist $numlabels $outlayer
    done
done