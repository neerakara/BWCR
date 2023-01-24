#!/bin/bash
anatomy='prostate'
dataset='RUNMC'
consloss=2
daprob=0.5
lamda=1.0
lamcons=0.01
heads=0
# if [[ "$method" -eq 0 ]]; then
#     heads=0
# elif [[ "$method" -eq 1 ]]; then
#     heads=0
# elif [[ "$method" -eq 10 ]]; then
#     heads=0
# elif [[ "$method" -eq 100 ]]; then
#     heads=0
# elif [[ "$method" -eq 2 ]]; then
#     heads=1
# elif [[ "$method" -eq 3 ]]; then
#     heads=1
# elif [[ "$method" -eq 20 ]]; then
#     heads=1
# elif [[ "$method" -eq 30 ]]; then
#     heads=1
# elif [[ "$method" -eq 200 ]]; then
#     heads=1
# elif [[ "$method" -eq 300 ]]; then
#     heads=1
# else
#     echo "I do not recognize this method."
# fi

for runnum in 2 3
do
    for cvfold in 1
    do
        for method in 200
        do
            for alpha in 100.0
            do
                sbatch /data/vision/polina/users/nkarani/projects/crael/seg/train.sh $anatomy $dataset $cvfold $runnum $daprob $heads $method $lamda $lamcons $consloss $alpha
            done
        done
    done
done