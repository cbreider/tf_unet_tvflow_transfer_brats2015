#!/bin/bash

# script for running k fold cross validation on pre trained model. All configuration should be done in configuration.py including the number of folds

#conda activate masterp36

cp configs/configuration_tv_regression_single.py configuration.py
python3 train.py --mode 1 --cuda_device 0 --name="TV_REG_Single_Scale" --create_new_split

cp configs/configuration_tv_regression_multi.py configuration.py
python3 train.py --mode 1 --cuda_device 0 --name="TV_REG_Multi_Scale"

cp configs/configuration_autoencoder.py configuration.py
python3 train.py --mode 5 --cuda_device 0 --name="Denoising_AE"

cp configs/configuration_autoencoder.py configuration.py
python3 train.py --mode 4 --cuda_device 0 --name="AE_Skip_Layers"

cp configs/configuration_autoencoder_wo_skip_layers.py configuration.py
python3 train.py --mode 4 --cuda_device 0 --name="AE_No_Skip_Layers"

cp configs/configuration_segmentation_single.py configuration.py
python3 train.py --mode 2 --cuda_device 0 --name="TV_SEG"


