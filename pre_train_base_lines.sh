#!/bin/bash

# script for running k fold cross validation on pre trained model. All configuration should be done in configuration.py including the number of folds

#conda activate masterp36

cp configs/configuration_skip_layer_autoencoder.py configuration.py
python3 train.py --mode 5 --cuda_device 0 --create_new_split --name="Skip_Layers"

cp configs/configuration_skip_layer_autoencoder.py configuration.py
python3 train.py --mode 4 --cuda_device 0 --name="No_Skip_Layers"

cp configs/configuration_tv_reg_single.py configuration.py
python3 train.py --mode 1 --cuda_device 0 --name="Single_Scale"

cp configs/configuration_tv_reg_multi.py configuration.py
python3 train.py --mode 1 --cuda_device 0 --name="Multi_Scale"




