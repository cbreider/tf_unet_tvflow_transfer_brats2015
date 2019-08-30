from src.utils.path_utils import DataPaths
from src.utils.split_utilities import TrainingDataset
from src.tf_data_pipeline_wrapper import ImageData
from src.tf_unet import unet
import src.utils.gpu_selector as cuda_selector
import src.trainer as trainer
import argparse
import configuration as config
import tensorflow as tf
import logging
import src.utils.data_utils as dutil
from src.utils.enum_params import TrainingModes, DataModes, Optimizer, RestoreMode

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--create_new_split",
                        help="create a new split rather than loading one (default False)",
                        action='store_true')
    parser.add_argument("--create_summaries",
                        help="crates a tensorboard summary (default true)",
                        action='store_true')
    parser.add_argument("--restore_path",
                        help="Path of a stored model. If given it will load this model",
                        type=str, default=None)
    parser.add_argument("--restore_mode",
                        help="Mode of restoring session. 1=Complete Session, 2=Without Out Layer, 3=Complete Net"
                        "Only used if restore_path is given",
                        type=int, default=1)
    parser.add_argument("--caffemodel_path",
                        help="Path of a preatrined caffe model (hfd5). If given it will load this model",
                        type=str, default=None)
    parser.add_argument('--cuda_device', type=int, default=-1,
                        help='Number of cuda device to use (optional)')

    args = parser.parse_args()

    create_new_training_split = False
    create_summaries = True
    restore_path = None
    caffemodel_path = None
    restore_mode = RestoreMode(args.restore_mode)

    if args.create_new_split:
        create_new_training_split = True
    if not args.create_summaries:
        create_summaries = False
    if args.restore_path is not None:
        restore_path = args.restore_path
    if args.caffemodel_path is not None:
        caffemodel_path = args.caffemodel_path
    if args.cuda_device >= 0:
        cuda_selector.set_cuda_gpu(args.cuda_device)

    # tf.enable_eager_execution()
    tf.reset_default_graph()

    data_paths = DataPaths(data_path="default", mode="SEGMENTATION")
    data_paths.load_data_paths()

    file_paths = TrainingDataset(paths=data_paths,
                                 mode=TrainingModes.SEGMENTATION,
                                 new_split=create_new_training_split,
                                 split_ratio=config.DataParams.split_train_val_ratio,
                                 nr_of_samples=config.DataParams.nr_of_samples,
                                 use_scale_as_gt=config.DataParams.use_scale_image_as_gt,
                                 load_only_mid_scans=config.DataParams.load_only_middle_scans,
                                 use_modalities=config.DataParams.use_modalities,
                                 use_mha=True)

    train_set = dutil.load_dataset_from_mha_files(file_paths.train_paths)
    training_data = ImageData(data=train_set,
                              batch_size=config.TrainingParams.batch_size_train,
                              buffer_size=config.TrainingParams.buffer_size_train,
                              shuffle=config.DataParams.shuffle,
                              mode=DataModes.TRAINING,
                              train_mode=TrainingModes.SEGMENTATION,
                              in_img_size=config.DataParams.raw_image_size,
                              set_img_size=config.DataParams.set_image_size,
                              data_max_value=config.DataParams.data_max_value,
                              data_norm_value=config.DataParams.norm_image_value,
                              crop_to_non_zero=config.DataParams.crop_to_non_zero,
                              do_augmentation=config.DataParams.do_image_augmentation,
                              normalize_std=config.DataParams.normailze_std,
                              nr_of_classes=config.DataParams.nr_of_classes_seg_mode,
                              nr_channels=config.DataParams.nr_of_channels)

    validation_set = dutil.load_dataset_from_mha_files(file_paths.validation_paths)
    validation_data = ImageData(data=validation_set,
                                batch_size=config.TrainingParams.buffer_size_val,
                                buffer_size=config.TrainingParams.buffer_size_val,
                                shuffle=config.DataParams.shuffle,
                                mode=DataModes.VALIDATION,
                                train_mode=TrainingModes.SEGMENTATION,
                                in_img_size=config.DataParams.raw_image_size,
                                set_img_size=config.DataParams.set_image_size,
                                data_max_value=config.DataParams.data_max_value,
                                data_norm_value=config.DataParams.norm_image_value,
                                crop_to_non_zero=config.DataParams.crop_to_non_zero_val,
                                do_augmentation=config.DataParams.do_image_augmentation_val,
                                normalize_std=config.DataParams.normailze_std,
                                nr_of_classes=config.DataParams.nr_of_classes_seg_mode,
                                nr_channels=config.DataParams.nr_of_channels)

    training_data.create()
    validation_data.create()

    net = unet.Unet(n_channels=config.DataParams.nr_of_channels,
                    n_class=config.DataParams.nr_of_classes_seg_mode,
                    cost_function=config.ConvNetParams.cost_function,
                    summaries=create_summaries,
                    class_weights=config.ConvNetParams.class_weights,
                    regularizer=config.ConvNetParams.regularizer,
                    n_layers=config.ConvNetParams.num_layers,
                    keep_prob=config.ConvNetParams.keep_prob_dopout,
                    features_root=config.ConvNetParams.feat_root,
                    filter_size=config.ConvNetParams.filter_size,
                    pool_size=config.ConvNetParams.pool_size,
                    freeze_down_layers=config.ConvNetParams.freeze_down_layers,
                    freeze_up_layers=config.ConvNetParams.freeze_up_layers,
                    use_padding=config.ConvNetParams.padding,
                    batch_norm=config.ConvNetParams.batch_normalization,
                    add_residual_layer=config.ConvNetParams.add_residual_layer,
                    use_scale_image_as_gt=config.DataParams.use_scale_image_as_gt,
                    act_func_out=config.ConvNetParams.activation_func_out)

    opt_args = None
    if config.TrainingParams.optimizer == Optimizer.MOMENTUM:
        opt_args = config.TrainingParams.momentum_args
    elif config.TrainingParams.optimizer == Optimizer.ADAGRAD:
        opt_args = config.TrainingParams.adagrad_args
    elif config.TrainingParams.optimizer == Optimizer.ADAM:
        opt_args = config.TrainingParams.adam_args

    trainer = trainer.Trainer(net=net,
                              norm_grads=config.TrainingParams.norm_grads,
                              optimizer=config.TrainingParams.optimizer,
                              opt_kwargs=opt_args)

    path = trainer.train(data_provider_train=training_data,
                         data_provider_val=validation_data,
                         out_path=data_paths.tf_out_path,
                         training_iters=config.TrainingParams.training_iters,
                         epochs=config.TrainingParams.num_epochs,
                         dropout=config.ConvNetParams.keep_prob_dopout,  # probability to keep units
                         display_step=config.TrainingParams.display_step,
                         write_graph=config.TrainingParams.write_graph,
                         restore_path=restore_path,
                         caffemodel_path=caffemodel_path,
                         restore_mode=restore_mode)