#!/bin/bash

# script for running k fold cross validation on pre trained model. All configuration should be done in configuration.py including the number of folds

#conda activate masterp36

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 1.0 --include_testing --create_new_split

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.9 --include_testing

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.8 --include_testing

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.7 --include_testing

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.6 --include_testing

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.5 --include_testing

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.4 --include_testing

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.3 --include_testing

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.2 --include_testing

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.15 --include_testing

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.1 --include_testing

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.05 --include_testing

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.025 --include_testing

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.0125 --include_testing