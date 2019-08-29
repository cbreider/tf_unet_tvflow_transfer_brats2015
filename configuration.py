"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019
"""


import numpy as np
from src.utils.enum_params import Optimizer, Cost


class DataParams:
    """ Data parameters"""
    # label values of BRATS2015 gt
    brats_label_values = [
        1,  # necrosis
        2,  # edema
        3,  # non-enhancing tumor
        4,  # enhancing tumor
        0  # everything else
    ]

    raw_data_height = 240
    raw_data_width = 240
    raw_image_size = [raw_data_height,
                      raw_data_width]   # size of the raw images
    set_data_height = 240
    set_data_width = 240
    set_image_size = [set_data_height,
                      set_data_width]  # size which the images should be reszied to for training
    nr_of_channels = 1              # number of channles of in/out images (grayscale)
    nr_of_classes_seg_mode = \
        len(brats_label_values)
    nr_of_classes_tv_flow_mode = 1  # one class for each channel of 8bit image
    shuffle = True                  # Set true to extra Shuffle Trining Data. Note dict.items() is allready random
    do_image_augmentation = True    # Set True to augment training images random crapp, flip, rotation
    do_image_augmentation_val = False  # Set True to augment training images random crapp, flip, rotation for validation
    split_train_val_ratio = 0.7     # Ration of Nr Training images to Val images
    use_scale_image_as_gt = False   # choose if you want to use tv scale image instead of smoothed (only training)
    crop_to_non_zero = True         # Choose True to alway crop Training images to region of non zero values
    crop_to_non_zero_val = False    # Choose True to alway crop Training images to region of non zero values for validation
    norm_image_value = 1.0          # Values which Images should be normed to during pre processing
    data_max_value = 255.0          # Max value of inout images (uint8)
    normailze_std = True            # normalize standard deviation for images during pre processing
    load_only_middle_scans = True   # load only slice 40 - 120
    nr_of_samples = 0               # use only a subset of images. if 0 all data is used


class ConvNetParams:
    """ ConvNetParams parameters"""
    num_layers = 5                  # number of encoder layers including bottom layer
    feat_root = 64                  # number of feature maps/kernels in the first layer
    filter_size = 3                 # kernel size
    pool_size = 2                   # size of max pooling
    keep_prob_dopout = 0.75          # keep prob for dropout
    cost_function = Cost.MSE        # Cost function to use. Choose from class Cost(Enum)
    padding = True                  # Use padding to preserve feature map size and prevent downscaling
    batch_normalization = True      # Use Batchnormalization Yes/No
    class_weights = None            # weight for each individual class # TODO ?
    regularizer = 0.00001           # lambda value for l2 regualizer
    add_residual_layer = True       # Add residual layer/skip layer at the end output = input + last_layer
    freeze_down_layers = True       # freeze encoder layers during training
    freeze_up_layers = False        # freeze decoder layers during training


class TrainingParams:
    """ Training parameters"""
    num_epochs = 100000             # number of training epochs
    label_smothing = 0              # smooth label values int gt to confuse network # TODO ?
    optimizer = Optimizer.ADAM      # Optimizer to use. Choose from class Optimizer(Enum):
    batch_size_train = 4            # batch size used for training
    batch_size_val = 32             # batch size used for validation
    buffer_size_train = 500         # buffer size for tf training data pipeline (only used for tv training)
    buffer_size_val = 500           # buffer size for tf validation data pipeline (only used for tv training)
    norm_grads = False              # norm gradients in summary
    training_iters = 1000           # iterations per epoch
    display_step = 200              # number of iterations between
    write_graph = True              # write graph in tf summary

    adam_args = dict(learning_rate=0.00001,  # Hyperparameters for Adam optimzer
                     beta1=0.9,
                     beta2=0.999,
                     epsilon=1e-08,
                     use_locking=False,
                     name='Adam',
                     decay_rate=0.95,
                     decay_steps=5000)
    momentum_args = dict(momentum=0.99,     # Hyperparameters for Momentum optimzer
                         learning_rate=0.00001,
                         decay_rate=0.95,
                         use_locking=False,
                         name='Momentum',
                         use_nesterov=False,
                         decay_steps=5000)
    adagrad_args = dict(learning_rate=0.001,  # Hyperparameters for Adagrd optimzer
                        initial_accumulator_value=0.1,
                        use_locking=False,
                        name='Adagrad')





