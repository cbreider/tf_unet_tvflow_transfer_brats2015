"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019
"""


from enum import Enum
import tensorflow as tf
import numpy as np


class Cost(Enum):
    """
    cost functions
    """
    CROSS_ENTROPY = 1
    DICE_COEFFICIENT = 2
    MEAN_SQUARED = 3


class DataModes(Enum):
    """
    running modes
    """
    TRAINING = 1
    VALIDATION = 2
    TESTING = 3


class TrainingModes(Enum):
    """
    training modes
    """
    TVFLOW = 1
    SEGMENTATION = 2


class Optimizer(Enum):
    """
    training optimzer
    """
    ADAGRAD = 1
    ADAM = 2
    MOMENTUM = 3


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
    seg_label_colors = np.array([
        [0,     0,   0],
        [255,   0,   0],
        [255, 255,   0],
        [0,   255,   0],
        [0,     0, 255]
    ])
    raw_data_height = 240  # height of training images
    raw_data_width = 240  # width of training images
    image_size = [raw_data_height, raw_data_width]
    set_data_height = 240
    set_data_width = 240  # width of training images
    set_image_size = [set_data_height, set_data_width]
    nr_of_channels = 1  # grayscale
    nr_of_classes_seg_mode = 5
    nr_of_classes_tv_flow_mode = 1 # one class for each channel of 8bit image
    shuffle = True  # dict.items() is allready random
    do_image_pre_processing = True  # only for training
    split_train_val_ratio = 0.7
    use_residual_as_gt = False
    data_type = tf.float16
    crop_to_non_zero = True
    norm_image_value = 1.0
    data_max_value = 255.0
    normailze_std = True
    load_only_middle_scans = True


class ConvNetParams:
    """ ConvNetParams parameters"""
    num_layers = 5
    feat_root = 64
    filter_size = 3
    pool_size = 2
    keep_prob_dopout = 0.5
    cost_function = Cost.MEAN_SQUARED
    padding = True
    batch_normalization = True
    class_weights = None  # TODO
    regularizer = 0.00001
    add_residual_layer = True


class TrainingParams:
    """ Training parameters"""
    num_epochs = 100000  # number of training epochs
    label_smothing = 0
    optimizer = Optimizer.ADAM
    batch_size_train = 4
    batch_size_val = 32
    buffer_size_train = 500
    buffer_size_val = 500
    norm_grads = False
    training_iters = 500
    display_step = 50
    write_graph = True
    adam_args = dict(learning_rate=0.0001,
                     beta1=0.9,
                     beta2=0.999,
                     epsilon=1e-08,
                     use_locking=False,
                     name='Adam',
                     decay_rate=0.95,
                     decay_steps=5000)
    momentum_args = dict(momentum=0.99,
                         learning_rate=0.00001,
                         decay_rate=0.95,
                         use_locking=False,
                         name='Momentum',
                         use_nesterov=False,
                         decay_steps=5000)
    adagrad_args = dict(learning_rate=0.001,
                        initial_accumulator_value=0.1,
                        use_locking=False,
                        name='Adagrad')

