#!/bin/bash
anatomy='prostate'
dataset='RUNMC'
method=3
daprob=0.5
lamcons=0.1

if [[ "$method" -eq 0 ]]; then
    heads=0
elif [[ "$method" -eq 1 ]]; then
    heads=0
elif [[ "$method" -eq 2 ]]; then
    heads=1
elif [[ "$method" -eq 3 ]]; then
    heads=1
else
    echo "I do not recognize this method."
fi

for cvfold in 1 2
do
    for runnum in 1 2 3
    do
        for alpha in 100.0
        do
            sbatch /data/vision/polina/users/nkarani/projects/crael/seg/train.sh $anatomy $dataset $cvfold $runnum $daprob $heads $method $lamcons $alpha
        done
    done
done