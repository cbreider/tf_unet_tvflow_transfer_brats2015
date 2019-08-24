"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019
"""


from enum import Enum
import tensorflow as tf


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
    seg_label_colors = [
        [255,   0,   0]
        [255, 255,   0]
        [0,   255,   0]
        [0,     0, 255]
        [0,     0,   0]
    ]
    raw_data_height = 240  # height of training images
    raw_data_width = 240  # width of training images
    image_size = [raw_data_height, raw_data_width]
    input_data_height = 572
    input_data_width = 572  # width of training images
    input_image_size = [input_data_height, input_data_width]
    output_data_height = 388
    output_data_width = 388 # width of training images
    output_image_size = [output_data_height, output_data_width]
    nr_of_channels = 1  # grayscale
    nr_of_classes_seg_mode = 4
    nr_of_classes_tv_flow_mode = 1 # one class for each channel of 8bit image
    shuffle = True  # dict.items() is allready random
    do_image_pre_processing = False  # only for training
    split_train_val_ratio = 0.7
    data_type = tf.float16


class ConvNetParams:
    """ ConvNetParams parameters"""
    num_layers = 5
    feat_root = 64
    filter_size = 3
    pool_size = 2
    keep_prob_dopout = 0.5
    cost_function = Cost.MEAN_SQUARED
    class_weights = None  # TODO
    regularizer = 0.001  # TODO


class TrainingParams:
    """ Training parameters"""
    num_epochs = 100000  # number of training epochs
    label_smothing = 0
    optimizer = Optimizer.MOMENTUM
    batch_size_train = 1
    batch_size_val = 32
    buffer_size_train = 500
    buffer_size_val = 500
    norm_grads = False
    training_iters = 100
    display_step = 10
    write_graph = True
    adam_args = dict(learning_rate=0.001,
                     beta1=0.9,
                     beta2=0.999,
                     epsilon=1e-08,
                     use_locking=False,
                     name='Adam')
    momentum_args = dict(momentum=0.99,
                         learning_rate=0.00001,
                         decay_rate=0.95,
                         use_locking=False,
                         name='Momentum',
                         use_nesterov=False)
    adagrad_args = dict(learning_rate=0.001,
                        initial_accumulator_value=0.1,
                        use_locking=False,
                        name='Adagrad')

