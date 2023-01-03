#!/bin/bash
test_anatomy='prostate'
method=3
daprob=0.5
lamda=1.0
lamcons=0.1

if [[ "$method" -eq 0 ]]; then
    heads=0
elif [[ "$method" -eq 1 ]]; then
    heads=0
elif [[ "$method" -eq 10 ]]; then
    heads=0
elif [[ "$method" -eq 2 ]]; then
    heads=1
elif [[ "$method" -eq 3 ]]; then
    heads=1
elif [[ "$method" -eq 20 ]]; then
    heads=1
elif [[ "$method" -eq 30 ]]; then
    heads=1
else
    echo "I do not recognize this method."
fi

for cvfold in 2
do
    for runnum in 1
    do
        for alpha in 100.0
        do
            for test_dataset in 'BMC'
            do
                sbatch /data/vision/polina/users/nkarani/projects/crael/seg/evaluate.sh $cvfold $runnum $test_anatomy $test_dataset $heads $method $lamda $lamcons $alpha
            done
        done
    done
done