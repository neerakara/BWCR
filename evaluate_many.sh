#!/bin/bash
test_anatomy='prostate'
test_dataset='BMC'
method=3
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

for cvfold in 1 2
do
    for runnum in 1 2 3
    do
        for alpha in 0.1 1.0 10.0 100.0
        do
            sbatch /data/vision/polina/users/nkarani/projects/crael/seg/evaluate.sh $cvfold $runnum $test_anatomy $test_dataset $heads $method $lamcons $alpha
        done
    done
done