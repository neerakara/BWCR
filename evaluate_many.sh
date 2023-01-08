#!/bin/bash
test_anatomy='placenta'
method=200
daprob=0.5
lamda=1.0
alpha=10.0
consloss=2

if [[ "$method" -eq 0 ]]; then
    heads=0
elif [[ "$method" -eq 1 ]]; then
    heads=0
elif [[ "$method" -eq 10 ]]; then
    heads=0
elif [[ "$method" -eq 100 ]]; then
    heads=0
elif [[ "$method" -eq 2 ]]; then
    heads=1
elif [[ "$method" -eq 3 ]]; then
    heads=1
elif [[ "$method" -eq 20 ]]; then
    heads=1
elif [[ "$method" -eq 30 ]]; then
    heads=1
elif [[ "$method" -eq 200 ]]; then
    heads=1
elif [[ "$method" -eq 300 ]]; then
    heads=1
else
    echo "I do not recognize this method."
fi

for cvfold in 4 5
do
    for runnum in 1 2 3 4 5
    do
        for lamcons in 0.001
        do
            for test_dataset in 'placenta'
            do
                sbatch /data/vision/polina/users/nkarani/projects/crael/seg/evaluate.sh $cvfold $runnum $test_anatomy $test_dataset $heads $method $lamda $lamcons $consloss $alpha
            done
        done
    done
done