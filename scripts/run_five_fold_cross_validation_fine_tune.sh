#!/bin/bash

# script for running k fold cross validation on pre trained model. All configuration should be done in configuration.py including the number of folds

#  conda activate masterp36


cp configs/configuration_brats_fine_tune_all_decoder.py configuration.py

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 2 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 1 --name="seg_all_rm_2_f1"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 2 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 2 --name="seg_all_rm_2_f2"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 2 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 3 --name="seg_all_rm_2_f3"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 2 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 4 --name="seg_all_rm_2_f4"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 2 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 5 --name="seg_all_rm_2_f5"


python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 4 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 1 --name="seg_all_rm_4_f1"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 4 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 2 --name="seg_all_rm_4_f2"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 4 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 3 --name="seg_all_rm_4_f3"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 4 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 4 --name="seg_all_rm_4_f4"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 4 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 5 --name="seg_all_rm_4_f5"


python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 8 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 1 --name="seg_all_rm_8_f1"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 8 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 2 --name="seg_all_rm_8_f2"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 8 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 3 --name="seg_all_rm_8_f3"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 8 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 4 --name="seg_all_rm_8_f4"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 8 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 5 --name="seg_all_rm_8_f5"


python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 16 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 1 --name="seg_all_rm_16_f1"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 16 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 2 --name="seg_all_rm_16_f2"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 16 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 3 --name="seg_all_rm_16_f3"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 16 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 4 --name="seg_all_rm_16_f4"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 16 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 5 --name="seg_all_rm_16_f5"


python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 24 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 1 --name="seg_all_rm_24_f1"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 24 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 2 --name="seg_all_rm_24_f2"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 24 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 3 --name="seg_all_rm_24_f3"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 24 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 4 --name="seg_all_rm_24_f4"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 24 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 5 --name="seg_all_rm_24_f5"


python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 32 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 1 --name="seg_all_rm_32_f1"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 32 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 2 --name="seg_all_rm_32_f2"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 32 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 3 --name="seg_all_rm_32_f3"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 32 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 4 --name="seg_all_rm_32_f4"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 32 --include_testing --restore_mode 2 --restore_path=/home/christian/Data_5/Projects/unet_brats2015/tf_model_output/TVFLOW_REGRESSION_ALL_multi_scale_2020-04-17_20:14:16 --take_fold_nr 5 --name="seg_all_rm_32_f5"

