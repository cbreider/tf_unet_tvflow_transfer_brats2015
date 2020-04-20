#!/bin/bash

# script for running k fold cross validation on pre trained model. All configuration should be done in configuration.py including the number of folds

#conda activate masterp36

cp configs/configuration_tv_regression_single.py configuration.py
python3 train.py --mode 1 --cuda_device 0 --name="Single_Scale" --create_new_split

cp configs/configuration_tv_regression_multi.py configuration.py
python3 train.py --mode 1 --cuda_device 0 --name="Multi_Scale"

cp configs/configuration_tv_segmentation_single.py configuration.py
python3 train.py --mode 2 --cuda_device 0 --name="Single_Scale" 

cp configs/configuration_tv_segmentation_multi.py configuration.py
python3 train.py --mode 2 --cuda_device 0 --name="Multi_Scale"

cp configs/configuration_autoencoder.py configuration.py
python3 train.py --mode 5 --cuda_device 0 --name="Skip_Layers"

cp configs/configuration_autoencoder.py configuration.py
python3 train.py --mode 4 --cuda_device 0 --name="Skip_Layers"

cp configs/configuration_autoencoder_wo_skip_layers.py configuration.py
python3 train.py --mode 4 --cuda_device 0 --name="No_Skip_Layers"




