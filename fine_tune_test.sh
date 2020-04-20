#!/bin/bash

# script for running k fold cross validation on pre trained model. All configuration should be done in configuration.py including the number of folds

#conda activate masterp36

cp configs/configuration1.py configuration.py
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.0125 --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_2020-04-07_23:24:35 --include_testing

cp configs/configuration2.py configuration.py
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.025 --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_2020-04-07_23:24:35 --include_testing

cp configs/configuration3.py configuration.py
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.1 --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_2020-04-07_23:24:35 --include_testing

cp configs/configuration4.py configuration.py
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 0.2 --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_2020-04-07_23:24:35 --include_testing




