#!/bin/bash

# script for running k fold cross validation on pre trained model. All configuration should be done in configuration.py including the number of folds

#  conda activate masterp36

for i in $(seq 1 "$num")
do
python3 train.py