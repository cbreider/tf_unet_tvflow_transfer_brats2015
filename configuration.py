"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019
"""


import src.utils.path_utils as p_util
from src.utils.enum_params import Optimizer, Cost, Activation_Func, TV_clustering_method, ConvNetType


class DataParams:
    """ Data parameters"""
    batch_size_train = 2            # batch size used for training
    batch_size_val = 1             # batch size used for validation
    buffer_size_train = 20         # buffer size for tf training data pipeline (only used for tv training)
    buffer_size_val = 64           # buffer size for tf validation data pipeline (only used for tv training)

    # label values of BRATS2015 gt
    brats_label_values = [
        1,  # necrosis
        2,  # edema
        3,  # non-enhancing tumor
        4,  # enhancing tumor
        0  # everything else
    ]
    modalities = ["mr_flair", "mr_t1", "mr_t1c", "mr_t2"]
    data_values = {modalities[0]: [9971, 59.5775, 217.8138],  # max, mean std of the training mri scans
                   modalities[1]: [11737, 79.6471, 272.0175],
                   modalities[2]: [11737, 81.787, 273.7927],
                   modalities[3]: [15281, 97.9302, 314.4067]}

    raw_data_height = 240
    raw_data_width = 240
    raw_image_size = [raw_data_height,
                      raw_data_width]   # size of the raw images
    set_data_height = 240
    set_data_width = 240
    set_image_size = [set_data_height,
                      set_data_width]  # size which the images should be reszied to for training

    nr_of_image_channels = 1              # number of channles of in/out images (grayscale)

     #img preprocessing and augmentation
    shuffle = True                  # Set true to extra Shuffle Trining Data. Note dict.items() is allready random
    do_image_augmentation_train = True    # Set True to augment training images random crapp, flip, rotation
    do_image_augmentation_val = True  # Set True to augment training images random crapp, flip, rotation for validation
    split_train_val_ratio = [0.99, 0.01] # [0.6, 0.2 0.2]     # Ration of Nr Training images to Val images (optioanl test)
    use_scale_image_as_gt = False   # choose if you want to use tv scale image instead of smoothed (only tv training)
    crop_to_non_zero_train = True         # Choose True to alway crop Training images to region of non zero values
    crop_to_non_zero_val = True    # Choose True to alway crop Training images to region of non zero values for validation
    norm_image_value = None          # Values which Images should be normed to during pre processing
    data_max_value = 255.0          # Max value of inout images (uint8)
    normailze_std = True            # normalize standard deviation for images during pre processing
    use_only_spatial_range = [30, 130]   # use only slices use_only_spatial_range[0] to use_only_spatial_range[1] because
                                        # it is unlikly to be tumot regions in the outer scans. use_only_spatial_range= None to use all scans
    nr_of_samples = 0               # use only a subset of images. if 0 all data is used
    use_modalities = [modalities[0], modalities[1], modalities[2], modalities[3]]  # modalities used for training
    nr_of_input_modalities = len(use_modalities)              # number of channles of in/out images (grayscale)
    nr_of_classes = 2
    use_mha_files_instead = False
    load_tv_from_file = False

    clustering_method = TV_clustering_method.STATIC_BINNING

    tv_and_clustering_params = dict(k_means_pre_cluster=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    tv_weight=0.1,   #params for tv smoothing
                                    tv_eps=0.00001,
                                    tv_tau=0.125,
                                    tv_m_itr=30,
                                    km_m_itr=100, # params for kmeans clustering nr of clusters = nr of classes.Only used im tv
                                                    # clustering with kmeans)
                                    ms_m_itr=-1,  # params for mean shift clustering used in tv training
                                    window_size=0.01,
                                    bin_seeding=True)


class ConvNetParams:
    """ ConvNetParams parameters"""
    conv_net_type = ConvNetType.U_NET_2D
    num_layers = 5                  # number of encoder layers including bottom layer
    feat_root = 64                  # number of feature maps/kernels in the first layer
    filter_size = 3                 # kernel size
    pool_size = 2                   # size of max pooling
    cost_function = Cost.CROSS_ENTROPY        # Cost function to use. Choose from class Cost(Enum)
    padding = True                  # Use padding to preserve feature map size and prevent downscaling
    batch_normalization = True      # Use Batchnormalization Yes/No
    class_weights = None #[1.0, 3.0]            # weight for each individual class # TODO ?
    regularizer = 0.0000001           # lambda value for l2 regualizer
    tv_regularizer = 0.01            # tv regularize for TV loss. oly used if Cost funcion is TV
    add_residual_layer = False       # Add residual layer/skip layer at the end output = input + last_layer
    freeze_down_layers = False       # freeze encoder layers during training
    freeze_up_layers = False        # freeze decoder layers during training
    activation_func_out = Activation_Func.RELU  # Act func for output map # noe for regression
    nr_input_channels = DataParams.nr_of_input_modalities
    nr_of_classes = DataParams.nr_of_classes
    use_scale_as_gt = DataParams.use_scale_image_as_gt
    max_tv_value = DataParams.norm_image_value
    two_classe_as_binary = True # treat a two class output[x1, x2] as binary classifiaction 0 or 1 for the optimization


class TrainingParams:
    """ Training parameters"""
    num_epochs = 100000             # number of training epochs
    training_iters = 15000           # iterations per epoch
    display_step = 1000              # number of iterations between
    label_smothing = 0              # smooth label values int gt to confuse network # TODO ?
    optimizer = Optimizer.ADAM      # Optimizer to use. Choose from class Optimizer(Enum):
    batch_size_train = DataParams.batch_size_train            # batch size used for training
    batch_size_val = DataParams.batch_size_val             # batch size used for validation
    buffer_size_train = DataParams.buffer_size_train # buffer size for tf training data pipeline (only used for tv training)
    buffer_size_val = DataParams.buffer_size_val # buffer size for tf validation data pipeline (only used for tv training)
    norm_grads = False              # norm gradients in summary
    write_graph = True              # write graph in tf summary
    keep_prob_dopout = 0.5          # keep prob for dropout

    adam_args = dict(learning_rate=0.0001,  # Hyperparameters for Adam optimzer
                     beta1=0.9,
                     beta2=0.999,
                     epsilon=1e-08,
                     use_locking=False,
                     name='Adam',
                     decay_rate=0.5,
                     decay_steps=30000)
    momentum_args = dict(momentum=0.99,     # Hyperparameters for Momentum optimzer
                         learning_rate=0.00001,
                         decay_rate=0.90,
                         use_locking=False,
                         name='Momentum',
                         use_nesterov=False,
                         decay_steps=10000)
    adagrad_args = dict(learning_rate=0.001,  # Hyperparameters for Adagrd optimzer
                        initial_accumulator_value=0.1,
                        use_locking=False,
                        name='Adagrad')





