"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""


from src.utils.enum_params import Optimizer, Cost, Activation_Func, TV_clustering_method, ConvNetType, Subtumral_Modes


class DataParams:
    """ Data parameters"""

    # batch size used for training
    batch_size_train = 16
    # batch size used for validation. Attention: Due to implementation only 1 is possible at the moment
    batch_size_val = 1
    # buffer size for tf training data pipeline
    buffer_size_train = 64
    # buffer size for tf validation data pipeline
    buffer_size_val = 20

    # label values of BRATS2015
    brats_label_values = [
        1,  # necrosis
        2,  # edema
        3,  # non-enhancing tumor
        4,  # enhancing tumor
        0  # everything else
    ]

    # mask for segmentation training: Complete, core or enhancing. Or ALL if all five classes should be separatly predicted
    segmentation_mask = Subtumral_Modes.ALL

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
                           10.0,  # sigma deformation magnitude
                           0.8]  # max zoom factor
    # normalize standard deviation for images during pre processing
    normailze_std = True
    # value to which images should be normed to during pre processing. If None original max vales are kept
    norm_max_image_value = None
    # Max value of input images (uint16)
    data_max_value = 65535.0
    # nr of folds for k fold cross validation
    nr_k_folds = 5
    # nr of validation samples taken from training set in k fold cross validtaion
    k_fold_nr_val_samples = 20
    # Ration of Nr iraining images to Val images (optioanl test) if new random split is created. Only used if not k fold
    # argumnet is passed (k_fold cross validation is not used). Must sum up to 1
    split_train_val_ratio = [0.8, 0.2]
    # split_train_val_ratio = [0.75, 0.05, 0.2]
    # use only a subset of training images. values from >0.0 - 1.0 (1.0 for all traing data)
    training_data_portion = 1.0
    # set True if pre computed tv images should be red from disc. If False tv is computed in data pipeline
    load_tv_from_file = False
    # choose if you want to use tv scale image instead of smoothed (only tv training and only if load_tv_from_file=True)
    use_scale_image_as_gt = False
    # use only slices use_only_spatial_range[0] to use_only_spatial_range[1] because it is unlikely to be tumor regions
    # in the outer scans. use_only_spatial_range=None to use all scans
    use_only_spatial_range = None #[30, 130]
    # modalities used for training
    use_modalities = [modalities[0], modalities[1], modalities[2], modalities[3]]
    # number of channels of generated input images (grayscale)
    nr_of_input_modalities = len(use_modalities) * nr_of_image_channels
    # nr of classes of segmentation map (binary for gt segmentation, more for tv segmentation)
    nr_of_classes = 5
    # do not use tf data pipeline. Load all images into RAM before. Not used, because eats up all memory
    use_mha_files_instead = False
    # modalities used and combined for tv. None for preset (COMPLETE = flair+T2, CORE=T1c, ENHANCING=T1)
    combine_modalities_for_tv = [modalities[0], modalities[1], modalities[2], modalities[3]]
    # method for clustering TV Images in TV segmentation mode (Static binning, Kmeans or mean shift)
    clustering_method = TV_clustering_method.STATIC_BINNING
    # params for differnt tv clustering methods (tv smoothing
    tv_and_clustering_params = dict(k_means_pre_cluster=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #not used
                                    # params for tv smoothing
                                    tv_weight=0.1, tv_eps=0.00001, tv_tau=0.125, tv_m_itr=50,
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
    # number of feature maps/kernels in the first layer (original 64)
    feat_root = 64
    # kernel size = filter_size x filter_size
    filter_size = 3
    # size of max pooling pool_size x pool_size
    pool_size = 2
    # Cost function to use. Choose from class Cost(Enum)
    cost_function = Cost.CROSS_ENTROPY
    # weighting if BATCH_DICE_SOFT_CE is chosen. loss = cost_weight * Dice_loss + (1-cost_weight) * cross_entropy_loss
    cost_weight = 0.8
    # Use padding to preserve feature map size and prevent downscaling
    padding = True
    # Use Batch normalization Yes/No
    batch_normalization = False
    # weight for each class if Cross Entropy loss is chosen. length must correspond to nr of classes.
    # None to not use any weighting
    class_weights = [0.01, 1.0, 1.0, 1.0, 1.0]
    # lambda value for l2 regualizer. Set None do not use l2 regularizer
    regularizer = None
    # tv regularize for TV loss. oly used if Cost funcion is TV
    tv_regularizer = 0.01
    # Add residual layer/skip layer at the end output = input + last_layer (only for tv regression). NOT useful
    add_residual_layer = False
    # freeze encoder layers during training
    freeze_down_layers = False
    # freeze decoder layers during training
    freeze_up_layers = False
    # Act func for output map. ATTENTION: Please choose none. actfunc is added prediction step
    # softmax multi class classifiavtion, sigmoid binary
    activation_func_out = Activation_Func.NONE
    # number of channels of generated input images (grayscale)
    nr_input_channels = DataParams.nr_of_input_modalities
    # nr of output classes
    nr_of_classes = DataParams.nr_of_classessss
    # choose if you want to use tv scale image instead of smoothed (only tv training and only if load_tv_from_file=True)
    use_scale_as_gt = DataParams.use_scale_image_as_gt
    # max value of TV images in regression
    max_tv_value = 1.0


class TrainingParams:
    """ Training parameters"""
    # batch size used for training
    batch_size_train = DataParams.batch_size_train
    # batch size used for validation
    batch_size_val = DataParams.batch_size_val
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
    num_epochs = 100
    # iterations per epoch
    training_iters = 2200
    # number of iterations between each
    display_step = 200
    # smooth label values int gt to confuse network. Not used  TODO ?
    label_smothing = 0
    # Optimizer to use. Choose from class Optimizer(Enum):
    optimizer = Optimizer.ADAM
    # keep prob for dropout
    keep_prob_dopout = 0.7
    # initial learning rate
    initial_learning_rate = 0.0001
    # store output images of validation
    store_val_images = True
    # store last feature maps  from cnn during validation ( only for middle scan)
    store_val_feature_maps = True
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
                     decay_steps=300000)
    # Hyperparameters for Momentum optimzer
    momentum_args = dict(momentum=0.99,
                         learning_rate=initial_learning_rate,
                         decay_rate=0.90,
                         use_locking=False,
                         name='Momentum',
                         use_nesterov=False,
                         decay_steps=10000)
    # Hyperparameters for Adagrd optimzer
    adagrad_args = dict(learning_rate=initial_learning_rate,
                        initial_accumulator_value=0.1,
                        use_locking=False,
                        name='Adagrad')





