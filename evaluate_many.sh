#!/bin/bash
test_anatomy='ms'
# test_anatomy='prostate'
method=200
daprob=0.5
lamda=1.0
lamcons=0.01
consloss=2
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

for cvfold in 2
do
    for runnum in 2
    do
        for alpha in 10.0
        do
            for test_dataset in 'OoD'
            # for test_dataset in 'RUNMC'
            do
                sbatch /data/vision/polina/users/nkarani/projects/crael/seg/evaluate.sh $cvfold $runnum $test_anatomy $test_dataset $heads $method $lamda $lamcons $consloss $alpha
            done
        done
    done
done