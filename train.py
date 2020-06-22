"""
Master Thesis Learning Feature Preserving Smoothing as Prior for Image Segmentation
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""


from src.utilities.path_utils import DataPaths
from src.utilities.split_utilities import TrainingDataset
from src.tf_data_pipeline_wrapper import ImageData
from src.tf_convnet import tf_convnet
import src.utilities.gpu_selector as cuda_selector
import src.trainer as trainer
import argparse
from configuration import Configuration as config
import tensorflow as tf
import logging
import src.utilities.logger as log
from src.utilities.enum_params import TrainingModes, DataModes, Optimizer, RestoreMode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        help="Mode of training session. 1=TV regression training, 2=TV clustering training, "
                             "3=BRATS Segmentation, 4=Auto Encoder, 5= Denoising Autoencoder",
                        type=int, default=1, required=True)
    parser.add_argument("--create_new_split",
                        help="create a new split rather than loading one (default False)",
                        action='store_true')
    parser.add_argument("--do_not_create_summaries",
                        help="crates a tensorboard summary (default false)",
                        action='store_true')
    parser.add_argument("--restore_path",
                        help="Path of a pretrained stored model. If given it will load this model",
                        type=str, default=None)
    parser.add_argument("--name",
                        help="name to include in the output path",
                        type=str, default="")
    parser.add_argument("--caffemodel_path",
                        help="Path of a pretrained caffe model (hfd5). If given it will load this model",
                        type=str, default=None)
    parser.add_argument('--cuda_device', type=int, default=-1,
                        help='Number of cuda device to use (optional)')
    parser.add_argument("--restore_mode",
                        help="Mode of restoring session. 1=Complete Session, 2=Without Out Layer, 3=Complete Net. "
                             "Only used if restore_path is given",
                        type=int, default=1)
    parser.add_argument('--take_fold_nr', type=int, default=0,
                        help='use fold with nr x from five fold split. If also --create_new_split is given a new '
                             'x fold is made')
    parser.add_argument('--include_testing', action='store_true',
                        help='Run evaluation on test set after optimization is finished')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the Brats training dataset. Files have to be 2D images and ordered in the same way'
                             '(HGG/LGG --> Patient --> modality--> *.png). Default ../dataset')
    parser.add_argument('--nr_training_scans', type=int, default=-1,
                        help='Use only a specific portion of all training samples')
    parser.add_argument('--reuse_out_folder', action='store_true',
                        help='Reuses the output folder of a model if it is restored')

    args = parser.parse_args()
    create_new_training_split = False
    create_summaries = True
    restore_path = None
    caffemodel_path = None
    train_mode = TrainingModes(args.mode)
    restore_mode = None
    include_testing = False
    data_path = "default"
    fold_nr = args.take_fold_nr
    name = ""
    reuse_out_folder = False
    if args.create_new_split:
        create_new_training_split = True
    if args.do_not_create_summaries:
        create_summaries = False
    if args.restore_path is not None:
        restore_path = args.restore_path
    if args.restore_mode:
        restore_mode = RestoreMode(args.restore_mode)
    if args.caffemodel_path is not None:
        caffemodel_path = args.caffemodel_path
    if args.cuda_device >= 0:
        cuda_selector.set_cuda_gpu(args.cuda_device)
    if args.include_testing:
        include_testing = True
    if args.data_path:
        data_path = args.data_path
    if args.name:
        name = args.name
    if args.nr_training_scans and args.nr_training_scans >= 0:
        config.nr_training_scans = args.nr_training_scans
    if restore_path and args.reuse_out_folder:
        reuse_out_folder = args.reuse_out_folder

    # tf.enable_eager_execution()
    tf.reset_default_graph()
    config.set_nr_classes(train_mode)
    data_paths = DataPaths(data_path=data_path, mode=train_mode.name,
                           tumor_mode=config.segmentation_mask.name, name=name)
    data_paths.load_data_paths(mkdirs=True,
                               restore_dir=restore_path if reuse_out_folder else None)

    log.init_logger(type="train", path=data_paths.tf_out_path)

    logging.info("Given Args: mode: {}, create_new_split: {}, restore_path: {},"
                 " restore_mode: {}, caffemodel_path: {}, cuda_dev: {}, fold nr: {}, testing: {}, data path: {}, "
                 "train data portion: {}".format(
        train_mode, create_new_training_split, restore_path, restore_mode, caffemodel_path, args.cuda_device, fold_nr,
        include_testing, data_path, config.nr_training_scans))

    logging.info("Training mode: {}".format(train_mode))

    logging.info("Modalities for training: {}".format(config.use_modalities))

    logging.info("Allocating '{}'".format(data_paths.tf_out_path))

    file_paths = TrainingDataset(paths=data_paths, mode=train_mode, data_config=config,
                                 new_split=create_new_training_split,
                                 is_five_fold=True if fold_nr > 0 else False,
                                 five_fold_idx=fold_nr, nr_of_folds=config.nr_k_folds,
                                 k_fold_nr_val_samples=config.k_fold_nr_val_samples)

    training_data = ImageData(data=file_paths.train_paths,
                              mode=DataModes.TRAINING,
                              train_mode=train_mode,
                              data_config=config)

    validation_data = ImageData(data=file_paths.validation_paths,
                                mode=DataModes.VALIDATION,
                                train_mode=train_mode,
                                data_config=config)
    training_data.create()
    validation_data.create()
    test_data = None
    if include_testing:
        try:
            file_paths_test = TrainingDataset(paths=data_paths, mode=train_mode, data_config=config, new_split=False,
                                              load_test_paths_only=True, is_five_fold=True if fold_nr > 0 else False,
                                              five_fold_idx=fold_nr)
            test_data = ImageData(data=file_paths_test.test_paths,
                                                    mode=DataModes.VALIDATION,
                                                    train_mode=train_mode,
                                                    data_config=config)
            test_data.create()
            logging.info("Loaded {} test samples".format(test_data.size))
        except:
            logging.ERROR("Failed to load test dataset. Skipping testing!!!")

    if restore_mode != RestoreMode.ONLY_GIVEN_VARS:
        config.restore_layers = None
    net = tf_convnet.ConvNetModel(convnet_config=config, mode=train_mode,
                                  create_summaries=create_summaries)
    opt_args = None
    if config.optimizer == Optimizer.MOMENTUM:
        opt_args = config.momentum_args
    elif config.optimizer == Optimizer.ADAGRAD:
        opt_args = config.adagrad_args
    elif config.optimizer == Optimizer.ADAM:
        opt_args = config.adam_args

    trainer = trainer.Trainer(net=net, data_provider_train=training_data, data_provider_val=validation_data,
                              out_path=data_paths.tf_out_path, train_config=config,
                              restore_path=restore_path, caffemodel_path=caffemodel_path,
                              restore_mode=restore_mode, mode=train_mode, data_provider_test=test_data, fold_nr=fold_nr)

    path = trainer.train()
