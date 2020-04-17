"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""


from src.utilities.enum_params import Optimizer, Cost, Activation_Func, TV_clustering_method, ConvNetType, Subtumral_Modes
import sys

gettrace = getattr(sys, 'gettrace', None)


class DataParams:
    """ Data parameters"""

    # batch size used for training
    batch_size_train = 8
    # buffer size for tf training data pipeline
    buffer_size_train = 32
    # buffer size for tf validation data pipeline
    buffer_size_val = 10

    # label values of BRATS2015
    brats_label_values = [
        1,  # necrosis
        2,  # edema
        3,  # non-enhancing tumor
        4,  # enhancing tumor
        0  # everything else
    ]

    # mask for segmentation training: Complete, core or enhancing. Or ALL if all five classes should be separatly predicted
    segmentation_mask = Subtumral_Modes.COMPLETE

    # key for the four different modalities
    modalities = ["mr_flair", "mr_t1", "mr_t1c", "mr_t2"]

    # values (pre computed) per modality of all mri (training set) scans:  [max, mean, variance]
    data_values = {modalities[0]: [9971.0, 373.4186, 436.8327],
                   modalities[1]: [11737.0, 498.3364, 514.9137],
                   modalities[2]: [11737.0, 512.1146, 560.1438],
                   modalities[3]: [15281.0, 609.6377, 507.4553]}
    # to norm every sclice by its own values uncomment this
    #data_values = {modalities[0]: [None, None, None],
    #               modalities[1]: [None, None, None],
    #               modalities[2]: [None, None, None],
    #               modalities[3]: [None, None, None]}

    # size of the raw images
    raw_data_height = 240
    raw_data_width = 240
    raw_image_size = [raw_data_height,
                      raw_data_width]

    # size which the images should be resized to for training
    set_data_height = 240
    set_data_width = 240
    set_image_size = [set_data_height,
                      set_data_width]

    # number of channles of in/out images (grayscale)
    nr_of_image_channels = 1

    """img preprocessing and augmentation"""

    # Set true to extra Shuffle Training Data. Note dict.items() is already random
    shuffle = True
    # Choose True to always crop training images to region of non zero values at first
    crop_to_non_zero_train = False
    # Choose True to always crop validation images to region of non zero values at first
    crop_to_non_zero_val = False
    # Set True to augment training images random crop, zoom, flip, rotation, distortion
    do_image_augmentation_train = True
    # Set True to augment validation images random crop, zoom, flip, rotation, distortion
    do_image_augmentation_val = False
    # parameters for image distortion
    image_disort_params = [[2, 3, 3],  # displacement vector [img dim, plane, heigh
                           25.0,  # sigma deformation magnitude
                           0.8]  # max zoom factor
    # normalize standard deviation for images during pre processing
    normailze_std = True
    # value to which images should be normed to during pre processing. If None original max vales are kept
    norm_max_image_value = None
    # value to which images should be normed to during pre processing. If None original max vales are kept
    norm_max_image_value_tv = 10.0
    # Max value of input images (uint16)
    data_max_value = 65535.0
    # nr of folds for k fold cross validation
    nr_k_folds = 5
    # nr of validation samples taken from training set in k fold cross validtaion
    k_fold_nr_val_samples = 10
    # Ration of Nr iraining images to Val images (optioanl test) if new random split is created. Only used if not k fold
    # argumnet is passed (k_fold cross validation is not used). Must sum up to 1
    #split_train_val_ratio = [0.75, 0.25]
    split_train_val_ratio = [0.6, 0.15, 0.25]
    # use only a subset of training images. values from >0.0 - 1.0 (1.0 for all traing data)
    training_data_portion = 1.0
    # set True if pre computed tv images should be red from disc. If False tv is computed in data pipeline
    load_tv_from_file = False
    # choose if you want to use tv scale image instead of smoothed (only tv training and only if load_tv_from_file=True)
    use_scale_image_as_gt = False
    # use only slices use_only_spatial_range[0] to use_only_spatial_range[1] because it is unlikely to be tumor regions
    # in the outer scans. use_only_spatial_range=None to use all scans
    use_only_spatial_range = None # [30, 130]
    # use only every x-th  non tumor slice. Set None to use all slices
    use_empty_slice_probability = 0.1
    # modalities used for training
    use_modalities = [modalities[0], modalities[1], modalities[2], modalities[3]]
    # number of channels of generated input images (grayscale)
    nr_of_input_modalities = len(use_modalities) * nr_of_image_channels
    # nr of classes of segmentation map (binary for gt segmentation, more for tv segmentation)
    nr_of_classes = 1
    # modalities used and combined for tv. None for preset (COMPLETE = flair+T2, CORE=T1c, ENHANCING=T1)
    combine_modalities_for_tv = [modalities[0], modalities[1], modalities[2], modalities[3]]
    # method for clustering TV Images in TV segmentation mode (Static binning, Kmeans or mean shift)
    clustering_method = TV_clustering_method.STATIC_BINNING
    # To train the network with multiple TV scales set a range for the tv_weight. During training the tv weight will be
    # selected uniformly from the range. Set to None to train only with a single scale set in the parameters below
    tv_multi_scale_range = None #[0.125, 1.125]
    tv_static_multi_scale = [0.2, 0.4, 0.6]
    #tv_multi_scale_range = None
    # params for differnt tv clustering methods (tv smoothing
    tv_and_clustering_params = dict(
        k_means_pre_cluster=[-0.5, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
                             0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
                                # params for tv smoothing
                                    tv_weight=0.75, tv_eps=0.00001, tv_tau=0.125, tv_m_itr=50,
                                    # params for kmeans nr of clusters =nr_of_clases.Only used im tv
                                    # clustering with kmeans)
                                    km_m_itr=100,
                                    # params for mean shift clustering used in tv training
                                    ms_m_itr=-1,
                                    window_size=0.01,
                                    bin_seeding=True)


class ConvNetParams:
    """ ConvNetParams parameters"""
    # type of CNN. In the moment only 2D unet available
    conv_net_type = ConvNetType.U_NET_2D
    # number of encoder layers including bottom layer (5 for original U-net)
    num_layers = 5
    if gettrace():
        num_layers = 2
    # number of feature maps/kernels in the first layer (original 64)
    feat_root = 64
    if gettrace():
        feat_roots = 16
    # kernel size = filter_size x filter_size
    filter_size = 3
    # size of max pooling pool_size x pool_size
    pool_size = 2
    # Cost function to use. Choose from class Cost(Enum)
    cost_function = Cost.BATCH_DICE_SOFT
    # weighting if BATCH_DICE_SOFT_CE is chosen. loss = cost_weight * Dice_loss + (1-cost_weight) * cross_entropy_loss
    cost_weight = 0.7
    # Use padding to preserve feature map size and prevent downscaling
    padding = True
    # Use Batch normalization Yes/No
    batch_normalization = False
    # weight for each class if Cross Entropy loss is chosen. length must correspond to nr of classes.
    # None to not use any weighting
    class_weights_ce = [0.1, 1.0, 1.0, 1.0, 1.0]
    # weight for each class if Dice loss is chosen. length must correspond to nr of classes.
    class_weights_dice = None  # [0.01, 1.0, 1.0, 1.0, 1.0]
    # lambda value for l2 regualizer. Set None do not use l2 regularizer
    lambda_l2_regularizer = None #0.000001
    # lambda value for l1 regualizer. Set None do not use l2 regularizer
    lambda_l1_regularizer = None #0.00000001
    # use spatial (channel dropout instead of single neuron dropout
    spatial_dropuout = True
    # tv regularize for TV loss. oly used if Cost funcion is TV
    tv_regularizer = 0.01
    # Add residual layer/skip layer at the end output = input + last_layer (only for tv regression). NOT useful
    add_residual_layer = False
    # freeze layers during training. Set None to train all layers
    trainable_layers = {"down_conv_0": True, "down_conv_1": True, "down_conv_2": True, "down_conv_3": True,
                        "down_conv_4": True,
                        # up_conv consists of transpose cond and two convolutions
                        "up_conv_3": [True, True], "up_conv_2": [True, True], "up_conv_1": [True, True], "up_conv_0": [True, True],
                        "classifier": True}
    # trainable_layers = None
    # freeze layers during training. Set None to train all layers
    restore_layers = {"down_conv_0": True, "down_conv_1": True, "down_conv_2": True, "down_conv_3": True,
                      "down_conv_4": True,
                      # up_conv consists of transpose cond and two convolutions
                      "up_conv_3": [True, True], "up_conv_2": [True, True], "up_conv_1": [True, True], "up_conv_0": [True, True],
                      "classifier": False}
    # trainable_layers = None
    # Act func for output map. ATTENTION: Please choose none. actfunc is added prediction step
    # softmax multi class classifiavtion, sigmoid binary
    activation_func_out = Activation_Func.NONE
    # number of channels of generated input images (grayscale)
    nr_input_channels = DataParams.nr_of_input_modalities
    # nr of output classes
    nr_of_classes = DataParams.nr_of_classes
    # choose if you want to use tv scale image instead of smoothed (only tv training and only if load_tv_from_file=True)
    use_scale_as_gt = DataParams.use_scale_image_as_gt
    # max value of TV images in regression
    max_tv_value = 1.0


class TrainingParams:
    """ Training parameters"""
    # batch size used for training
    batch_size_train = DataParams.batch_size_train
    # buffer size for tf training data pipeline
    buffer_size_train = DataParams.buffer_size_train
    # buffer size for tf validation data pipeline
    buffer_size_val = DataParams.buffer_size_val
    # norm gradients in summary
    norm_grads = False
    # write graph in tf summary
    write_graph = True
    # log (to terminal) mini batch stats after training_iters. If False only average is logged
    log_mini_batch_stats = False
    # number of training epochs
    num_epochs = 15
    # iterations per epoch
    training_iters = 1000
    # number of iterations between each
    display_step = 100
    # smooth label values int gt to confuse network. Not used  TODO ?
    label_smothing = 0
    # Optimizer to use. Choose from class Optimizer(Enum):
    optimizer = Optimizer.ADAM
    # dropout probability for the first convolution in each block.
    # Note: it's unusual to use dropout in convolutional layers
    # but they did it in the original tf_unet implementation, so at least the option will be provided here.
    dropout_rate_conv1 = 0.2
    # dropout probability for the second convolution in each block
    dropout_rate_conv2 = 0.15
    # dropout_rate for the pooling and  layers
    dropout_rate_pool = 0.0
    # dropout_rate for the deconvolutional layers
    dropout_rate_tconv = 0.0
    # dropout_rate for the deconvolutional layers
    dropout_rate_concat = 0.0
    # initial learning rate
    initial_learning_rate = 0.00001
    # store output images of validation
    store_val_images = False
    # store last feature maps  from cnn during validation ( only for middle scan)
    store_val_feature_maps = False
    if gettrace():
        store_val_feature_maps = False
        store_val_images = False
    # stop training if validation loss has not decreased over last three epochs
    early_stopping = False

    # Hyperparameters for Adam optimzer
    adam_args = dict(learning_rate=initial_learning_rate,
                     beta1=0.9,
                     beta2=0.999,
                     epsilon=1e-08,
                     use_locking=False,
                     name='Adam',
                     decay_rate=0.1,
                     decay_steps=10000)
    # Hyperparameters for Momentum wawddwawoptimzer
    momentum_args = dict(momentum=0.99,
                         learning_rate=initial_learning_rate,
                         decay_rate=0.9,
                         use_locking=False,
                         name='Momentum',
                         use_nesterov=False,
                         decay_steps=10000)
    # Hyperparameters for Adagrd optimzer
    adagrad_args = dict(learning_rate=initial_learning_rate,
                        initial_accumulator_value=0.1,
                        use_locking=False,
                        name='Adagrad')







