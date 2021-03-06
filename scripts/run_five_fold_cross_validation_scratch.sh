#!/bin/bash

# script for running k fold cross validation from scratch. All configuration should be done in configuration.py including the number of folds

#  conda activate masterp36

cp configs/configuration_brats_scratch_all.py configuration.py
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 2 --include_testing --take_fold_nr 1 --name="scratch_2_f1" --create_new_split
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 2 --include_testing --take_fold_nr 2 --name="scratch_2_f2"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 2 --include_testing --take_fold_nr 3 --name="scratch_2_f3"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 2 --include_testing --take_fold_nr 4 --name="scratch_2_f4"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 2 --include_testing --take_fold_nr 5 --name="scratch_2_f5"

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 4 --include_testing --take_fold_nr 1 --name="scratch_4_f1"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 4 --include_testing --take_fold_nr 2 --name="scratch_4_f2"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 4 --include_testing --take_fold_nr 3 --name="scratch_4_f3"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 4 --include_testing --take_fold_nr 4 --name="scratch_4_f4"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 4 --include_testing --take_fold_nr 5 --name="scratch_4_f5"

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 8 --include_testing --take_fold_nr 1 --name="scratch_8_f1"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 8 --include_testing --take_fold_nr 2 --name="scratch_8_f2"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 8 --include_testing --take_fold_nr 3 --name="scratch_8_f3"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 8 --include_testing --take_fold_nr 4 --name="scratch_8_f4"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 8 --include_testing --take_fold_nr 5 --name="scratch_8_f5"


python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 16 --include_testing --take_fold_nr 1 --name="scratch_16_f1"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 16 --include_testing --take_fold_nr 2 --name="scratch_16_f2"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 16 --include_testing --take_fold_nr 3 --name="scratch_16_f3"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 16 --include_testing --take_fold_nr 4 --name="scratch_16_f4"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 16 --include_testing --take_fold_nr 5 --name="scratch_16_f5"

python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 24 --include_testing --take_fold_nr 1 --name="scratch_24_f1"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 24 --include_testing --take_fold_nr 2 --name="scratch_24_f2"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 24 --include_testing --take_fold_nr 3 --name="scratch_24_f3"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 24 --include_testing --take_fold_nr 4 --name="scratch_24_f4"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 24 --include_testing --take_fold_nr 5 --name="scratch_24_f5"


python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 32 --include_testing --take_fold_nr 1 --name="scratch_32_f1"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 32 --include_testing --take_fold_nr 2 --name="scratch_32_f2"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 32 --include_testing --take_fold_nr 3 --name="scratch_32_f3"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 32 --include_testing --take_fold_nr 4 --name="scratch_32_f4"
python3 train.py --mode 3 --cuda_device 0 --nr_training_scans 32 --include_testing --take_fold_nr 5 --name="scratch_32_f5"


#scratch with TV pseudo patient
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 2 --include_testing --take_fold_nr 1 --name=""scratch_pseudo_pateint_2_f1"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 2 --include_testing --take_fold_nr 2 --name=""scratch_pseudo_pateint_2_f2"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 2 --include_testing --take_fold_nr 3 --name=""scratch_pseudo_pateint_2_f3"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 2 --include_testing --take_fold_nr 4 --name=""scratch_pseudo_pateint_2_f4"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 2 --include_testing --take_fold_nr 5 --name=""scratch_pseudo_pateint_2_f5"

python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 4 --include_testing --take_fold_nr 1 --name=""scratch_pseudo_pateint_4_f1"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 4 --include_testing --take_fold_nr 2 --name=""scratch_pseudo_pateint_4_f2"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 4 --include_testing --take_fold_nr 3 --name=""scratch_pseudo_pateint_4_f3"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 4 --include_testing --take_fold_nr 4 --name=""scratch_pseudo_pateint_4_f4"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 4 --include_testing --take_fold_nr 5 --name=""scratch_pseudo_pateint_4_f5"

python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 8 --include_testing --take_fold_nr 1 --name=""scratch_pseudo_pateint_8_f1"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 8 --include_testing --take_fold_nr 2 --name=""scratch_pseudo_pateint_8_f2"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 8 --include_testing --take_fold_nr 3 --name=""scratch_pseudo_pateint_8_f3"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 8 --include_testing --take_fold_nr 4 --name=""scratch_pseudo_pateint_8_f4"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 8 --include_testing --take_fold_nr 5 --name=""scratch_pseudo_pateint_8_f5"


python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 16 --include_testing --take_fold_nr 1 --name=""scratch_pseudo_pateint_16_f1"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 16 --include_testing --take_fold_nr 2 --name=""scratch_pseudo_pateint_16_f2"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 16 --include_testing --take_fold_nr 3 --name=""scratch_pseudo_pateint_16_f3"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 16 --include_testing --take_fold_nr 4 --name=""scratch_pseudo_pateint_16_f4"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 16 --include_testing --take_fold_nr 5 --name=""scratch_pseudo_pateint_16_f5"

python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 24 --include_testing --take_fold_nr 1 --name=""scratch_pseudo_pateint_24_f1"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 24 --include_testing --take_fold_nr 2 --name=""scratch_pseudo_pateint_24_f2"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 24 --include_testing --take_fold_nr 3 --name=""scratch_pseudo_pateint_24_f3"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 24 --include_testing --take_fold_nr 4 --name=""scratch_pseudo_pateint_24_f4"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 24 --include_testing --take_fold_nr 5 --name=""scratch_pseudo_pateint_24_f5"


python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 32 --include_testing --take_fold_nr 1 --name=""scratch_pseudo_pateint_32_f1"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 32 --include_testing --take_fold_nr 2 --name=""scratch_pseudo_pateint_32_f2"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 32 --include_testing --take_fold_nr 3 --name=""scratch_pseudo_pateint_32_f3"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 32 --include_testing --take_fold_nr 4 --name=""scratch_pseudo_pateint_32_f4"
python3 train.py --mode 6 --cuda_device 0 --nr_training_scans 32 --include_testing --take_fold_nr 5 --name=""scratch_pseudo_pateint_32_f5"