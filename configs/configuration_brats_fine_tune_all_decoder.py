"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""


from src.utilities.enum_params import Optimizer, Cost, TV_clustering_method, ConvNetType, Subtumral_Modes, TrainingModes
import sys

gettrace = getattr(sys, 'gettrace', None)


class Configuration:
    """
    Configurable Parameters passed to other classes
    """

    """
    --------------------------------------------------------------------------------------------------------------------
    Data related Parameters for Brats 2015 data set
    --------------------------------------------------------------------------------------------------------------------
    """
    batch_size_train = 8
    """batch_size_train log string"""

    buffer_size_train = 32
    """buffer size for tf training data pipeline"""

    buffer_size_val = 10
    """buffer size for tf validation data pipeline"""

    brats_label_values = [
        1,  # necrosis
        2,  # edema
        3,  # non-enhancing tumor
        4,  # enhancing tumor
        0  # everything else
    ]
    """label values of BRATS2015"""

    segmentation_mask = Subtumral_Modes.ALL
    """ mask for segmentation training: Complete, core or enhancing.  
    Or ALL if all five classes should be separatly predicted"""

    modalities = ["mr_flair", "mr_t1", "mr_t1c", "mr_t2"]
    """key for the four different modalities"""

    data_values = {modalities[0]: [9971.0, 373.4186, 436.8327],
                   modalities[1]: [11737.0, 498.3364, 514.9137],
                   modalities[2]: [11737.0, 512.1146, 560.1438],
                   modalities[3]: [15281.0, 609.6377, 507.4553]}
    # to norm every sclice by its own values uncomment this
    #data_values = {modalities[0]: [None, None, None],
    #               modalities[1]: [None, None, None],
    #               modalities[2]: [None, None, None],
    #               modalities[3]: [None, None, None]}
    """" values (pre computed) per modality of all mri (training set) scans:  [max, mean, variance]
    per default pre computed vales per scan from the scan folder will be loaded. "data_values" will only be used if
    the values file ("values.json" could not be found"""

    raw_data_height = 240
    raw_data_width = 240
    raw_image_size = [raw_data_height,
                      raw_data_width]
    """size of the raw images"""

    set_data_height = 240
    set_data_width = 240
    set_image_size = [set_data_height,
                      set_data_width]
    """size which the images should be resized to for training"""

    nr_of_image_channels = 1
    """number of channles of in/out images (grayscale)"""

    data_max_value = 65535.0
    """Max value of input images (uint16)"""

    use_modalities = [modalities[0], modalities[1], modalities[2], modalities[3]]
    """modalities used for training"""

    """
    --------------------------------------------------------------------------------------------------------------------
    Parameters for creating split an loading data
    --------------------------------------------------------------------------------------------------------------------
    """
    nr_k_folds = 5
    """r of folds for k fold cross validation"""

    k_fold_nr_val_samples = 50
    """nr of validation samples taken from training set in k fold cross validation"""

    split_train_val_ratio = [0.6, 0.15, 0.25]
    """Ration of Nr iraining images to Val images (optional test)
    if new random split is created. Only used if not k fold
    argument is passed (k_fold cross validation is not used). Must sum up to 1
    split_train_val_ratio = [0.75, 0.25]"""

    nr_training_scans = -1
    """use only a subset of training images. (-1 for all training data)"""

    load_tv_from_file = False
    """set True if pre computed tv images should be red from disc. If False tv is computed in data pipeline"""

    use_scale_image_as_gt = False
    """choose if you want to use tv scale image instead of smoothed
    (only tv training and only if load_tv_from_file=True)"""

    use_only_spatial_range = None # [30, 130]
    """use only slices use_only_spatial_range[0] to use_only_spatial_range[1]
    because it is unlikely to be tumor regions
    in the outer scans. use_only_spatial_range=None to use all scans"""

    use_empty_slice_probability = 0.1
    """use only every x-th  non tumor slice. Set None to use all slices Not combinable with use_only_spatial_range"""

    """
    --------------------------------------------------------------------------------------------------------------------
    Parameters for Total Variation smoothing adn clustering/binning
    --------------------------------------------------------------------------------------------------------------------
    """
    norm_max_image_value_tv = None
    """value to which images should be normed to during pre processing. If None original max vales are kept. 
    must be given if training mode is TV Segmentation"""

    use_modalities_for_tv = [modalities[0], modalities[1], modalities[2], modalities[3]]
    """modalities used and combined for tv. None for preset (COMPLETE = flair+T2, CORE=T1c, ENHANCING=T1)"""

    clustering_method = TV_clustering_method.STATIC_BINNING
    """method for clustering TV Images in TV segmentation mode (Static binning, Kmeans or mean shift)"""

    static_cluster_centers = [-0.5, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
                              0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    """Static clusters used if TV clusteirng mode is set to STATIC_CLUSTERS"""

    nr_clusters = 8 if clustering_method != TV_clustering_method.STATIC_CLUSTERS else static_cluster_centers
    """Nr of clusters used for TV Segmentation"""

    tv_static_multi_scale = [0.2, 0.4, 0.6, 0.8]
    """Train the network with multiple sclaes each with a seperate output map for the network"""

    tv_multi_scale_range = None #[0.125, 1.125]
    """To train the network with multiple TV scales set a range for the tv_weight. During training the tv weight will be
    selected uniformly from the range. Set to None to train only with a single scale set in the parameters below"""

    tv_weight = 0.5
    """tv weight for single scale"""

    tv_eps = 0.00001
    """eps to stop tv after error reached eps* init_error"""

    tv_tau = 0.125
    """tv step size"""

    tv_m_itr = 50
    """ max number of iterations for tv"""

    kmeans_m_iter = 100
    """max number of iterations for k-means"""

    meanshift_m_iter = -1
    """ max number of iterations for mean shift. -1 for iterate till convergence"""

    meanshift_window_size = 0.01
    """windows size for mean shift"""

    meanshift_bin_seeding = True
    """use bin seeding for initial mean shift clusters to speed up algorithm"""

    """
    --------------------------------------------------------------------------------------------------------------------
    Pre Processing and Data Augmentation parameters
    --------------------------------------------------------------------------------------------------------------------
    """
    crop_to_non_zero_train = False
    """Choose True to always crop training images to region of non zero values at first"""

    crop_to_non_zero_val = False
    """Choose True to always crop validation images to region of non zero values at first"""

    do_image_augmentation_train = True
    """Set True to augment training images random crop, zoom, flip, rotation, distortion"""

    do_image_augmentation_val = False
    """Set True to augment validation images random crop, zoom, flip, rotation, distortion"""

    image_disort_params = [[2, 3, 3],  # displacement vector [img dim, plane, height]
                           25.0,  # sigma deformation magnitude
                           0.8]  # max zoom factor
    """parameters for image distortion"""

    normailze_std = True
    """normalize standard deviation for images during pre processing"""

    norm_max_image_value = None
    """value to which images should be normed to during pre processing. If None original max vales are kept"""

    """
    --------------------------------------------------------------------------------------------------------------------
    Class Parameters
    --------------------------------------------------------------------------------------------------------------------
    """
    nr_input_channels = len(use_modalities) * nr_of_image_channels
    """number of channels of generated input images (grayscale)"""

    nr_of_classes_brats_all = len(brats_label_values)
    """nr of classes of segmentation map for brats training on all classes"""

    nr_classes_brats_complete = 1
    """nr of classes for brats training on only one mask"""

    nr_classes_tv_regression = len(use_modalities_for_tv) * len(tv_static_multi_scale) if tv_static_multi_scale else 1
    """Nr of classes for TV regression mode"""

    nr_classes_tv_segmentation = (len(use_modalities_for_tv) * len(tv_static_multi_scale)
                                  if tv_static_multi_scale else 1) * nr_clusters
    """Nr of classes for TV segmentation mode"""

    nr_classes = -1
    """Nr of classes for the network. Depending on the training mode. Set in set_nr_classes"""

    def set_nr_classes(train_mode):
        """
        Sets the number of output classes (nr_classes) for the network depending on the training mode.
        Must be called before passes to the network

        :param train_mode: Training mode of the Network
        :type: TrainingMode
        """
        if train_mode == TrainingModes.BRATS_SEGMENTATION:
            if Configuration.segmentation_mask == Subtumral_Modes.ALL:
                Configuration.nr_classes = Configuration.nr_of_classes_brats_all
            else:
                Configuration.nr_classes = Configuration.nr_classes_brats_complete
        elif train_mode == TrainingModes.TVFLOW_REGRESSION:
            Configuration.nr_classes = Configuration.nr_classes_tv_regression
        elif train_mode == TrainingModes.BRATS_SEGMENTATION:
            Configuration.nr_classes = Configuration.nr_classes_tv_segmentation
        elif train_mode == TrainingModes.AUTO_ENCODER or train_mode == TrainingModes.DENOISING_AUTOENCODER:
            Configuration.nr_classes = len(Configuration.use_modalities)

    """
    --------------------------------------------------------------------------------------------------------------------
    ConvNetParams parameters
    --------------------------------------------------------------------------------------------------------------------
    """

    conv_net_type = ConvNetType.U_NET_2D
    """type of CNN. In the moment only 2D unet available"""

    num_layers = 5
    """Nr of encoder layers including bottom layer (5 for original U-net)"""
    if gettrace():
        num_layers = 2

    feat_root = 64
    """number of feature maps/kernels in the first layer (original 64)"""
    if gettrace():
        feat_roots = 16

    filter_size = 3
    """kernel size = filter_size x filter_size"""

    pool_size = 2
    """size of max pooling pool_size x pool_size"""

    cost_function = Cost.BATCH_DICE_SOFT
    """Cost function to use. Choose from class Cost(Enum)"""

    cost_weight = 0.7
    """weighting if BATCH_DICE_SOFT_CE is chosen. 
    loss = cost_weight * Dice_loss + (1-cost_weight) * cross_entropy_loss"""

    padding = True
    """Use padding to preserve feature map size and prevent downscaling"""

    class_weights_ce = None #[0.1, 1.0, 1.0, 1.0, 1.0]
    """weight for each class if Cross Entropy loss is chosen. length must correspond to nr of classes.
    None to not use any weighting"""

    class_weights_dice = None  # [0.01, 1.0, 1.0, 1.0, 1.0]
    """weight for each class if Dice loss is chosen. length must correspond to nr of classes."""

    tv_regularizer = 0.01
    """tv regularize for TV loss. oly used if Cost funcion is TV"""

    add_residual_layer = False
    """Add residual layer/skip layer at the end output = input + last_layer (only for tv regression). NOT useful"""

    remove_skip_layers = False
    """remove skip layer connections"""

    trainable_layers = dict(
        down_conv_3=False,      down_conv_2=False,      down_conv_1=False,        down_conv_0=False,
        down_conv_4=False,
        # up_conv consists of transpose cond and two convolutions
        up_conv_3=[True, True], up_conv_2=[True, True], up_conv_1=[True, True], up_conv_0=[True, True],
        classifier=True)
    # trainable_layers = None
    """freeze layers during training. Set None to train all layers"""


    restore_layers = dict(
        down_conv_3=True,      down_conv_2=True,      down_conv_1=True,        down_conv_0=True,
        down_conv_4=True,
        # up_conv consists of transpose cond and two convolutions
        up_conv_3=[True, True], up_conv_2=[True, True], up_conv_1=[True, True], up_conv_0=[True, True],
        classifier=False)
    # trainable_layers = None
    """ freeze layers during training. Set None to train all layers"""

    """
    --------------------------------------------------------------------------------------------------------------------
    Regularization Parameters
    --------------------------------------------------------------------------------------------------------------------
    """

    lambda_l2_regularizer = None
    # lambda_l2_regularizer = 0.000001
    """lambda value for l2 regularizer. Set None do not use l2 regularizer"""

    lambda_l1_regularizer = None
    # lambda_l1_regularizer = 0.00000001
    """lambda value for l1 regularizer. Set None do not use l2 regularizer"""

    spatial_dropuout = True
    """use spatial (channel dropout instead of single neuron dropout"""

    dropout_rate_conv1 = 0.0
    """dropout probability for the first convolution in each block.
    Note: it's unusual to use dropout in convolutional layers
    but they did it in the original tf_unet implementation, so at least the option will be provided here."""

    dropout_rate_conv2 = 0.0
    """dropout probability for the second convolution in each block"""

    dropout_rate_pool = 0.0
    """dropout_rate for the pooling and  layers"""

    dropout_rate_tconv = 0.0
    """dropout_rate for the deconvolutional layers"""

    dropout_rate_concat = 0.0
    """dropout_rate for the concating layers"""

    batch_normalization = False
    """ Use Batch normalization after each convolution Yes/No"""

    """
    --------------------------------------------------------------------------------------------------------------------
    Training parameters
    --------------------------------------------------------------------------------------------------------------------
    """
    num_epochs = 15
    """ number of training epochs. Note one epoch normaly represents a complete loop through the training data.
    But in this case we deal with very diffent number of training sets. So the number of iterations per epoch will be
    kept fixed"""

    training_iters = 1000
    """iterations per epoch"""

    display_step = 100
    """number of iterations between each"""

    label_smothing = 0
    """smooth label values int gt to confuse network. Not used """ #TODO?

    optimizer = Optimizer.ADAM
    """Optimizer to use. Choose from class Optimizer(Enum)"""

    initial_learning_rate = 0.00001
    """initial learning rate"""

    early_stopping_epochs = 3
    """stop training if validation loss has not decreased over the given epochs. Set None to not use early stopping"""

    unfreeze_all_layers_epochs = 3 if trainable_layers else -1
    """unfreeze al frozen layers has not decreased over the given epochs. Set -1 to not use """

    adam_args = dict(learning_rate=initial_learning_rate,
                     beta1=0.9,
                     beta2=0.999,
                     epsilon=1e-08,
                     use_locking=False,
                     name='Adam',
                     decay_rate=0.97,
                     decay_steps=1000)
    """Hyperparameters for Adam optimzer"""

    momentum_args = dict(momentum=0.99,
                         learning_rate=initial_learning_rate,
                         decay_rate=0.9,
                         use_locking=False,
                         name='Momentum',
                         use_nesterov=False,
                         decay_steps=10000)
    """Hyperparameters for Momentum optimzer"""

    adagrad_args = dict(learning_rate=initial_learning_rate,
                        initial_accumulator_value=0.1,
                        use_locking=False,
                        name='Adagrad')
    """Hyperparameters for Adagrd optimzer"""

    store_val_images = False
    """store output images of validation"""

    store_val_feature_maps = False
    """store last feature maps  from cnn during validation ( only for middle scan)"""
    if gettrace():
        store_val_feature_maps = False
        store_val_images = False

    norm_grads = False
    """norm gradients in summary"""

    write_graph = True
    """write graph in tf summary"""

    log_mini_batch_stats = False
    """log (to terminal) mini batch stats after training_iters. If False only average is logged"""

