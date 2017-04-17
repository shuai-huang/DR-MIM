#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/OpenBLAS/lib/
export OMP_NUM_THREADS=10

train_data=$1
test_data=$2
opt=$3
lab_train=$4
train_red=$5
test_red=$6
tsf_mat=$7

./main $train_data $test_data $opt $lab_train $train_red $test_red $tsf_mat
