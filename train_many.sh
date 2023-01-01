#!/bin/bash
method=1
daprob=0.5
lamcons=1.0

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

for cvfold in 1 2 3
do
    for runnum in 2 3
    do
        for alpha in 1.0
        do
            sbatch /data/vision/polina/users/nkarani/projects/crael/seg/train.sh $cvfold $runnum $daprob $heads $method $lamcons $alpha
        done
    done
done