from src.utils.path_utils import DataPaths
from src.utils.split_utilities import TrainingDataset
from src.tf_data_pipeline_wrapper import ImageData
from src.tf_unet import unet
import src.utils.gpu_selector as cuda_selector
import src.trainer as trainer
import argparse
import src.configuration as config
import tensorflow as tf
import logging


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
    parser.add_argument("--caffemodel_path",
                        help="Path of a preatrined caffe model (hfd5). If given it will load this model",
                        type=str, default=None)
    parser.add_argument("--freeze",
                        help="Freeze all layers except for last one",
                        action='store_true')
    parser.add_argument('--cuda_device', type=int, default=-1,
                        help='Number of cuda device to use (optional)')
    args = parser.parse_args()
    create_new_training_split = False
    create_summaries = True
    restore_path = None
    freeze = False
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
    if args.freeze:
        freeze = True

    # tf.enable_eager_execution()
    tf.reset_default_graph()

    data_paths = DataPaths(data_path="default", mode="SEGMENTATION")
    data_paths.load_data_paths()

    file_paths = TrainingDataset(paths=data_paths,
                                 mode=config.TrainingModes.SEGMENTATION,
                                 new_split=create_new_training_split,
                                 split_ratio=config.DataParams.split_train_val_ratio)

    training_data = ImageData(file_paths=file_paths.train_paths,
                              batch_size=config.TrainingParams.batch_size_train,
                              buffer_size=config.TrainingParams.buffer_size_train,
                              shuffle=config.DataParams.shuffle,
                              do_pre_processing=config.DataParams.do_image_pre_processing,
                              mode=config.DataModes.TRAINING,
                              train_mode=config.TrainingModes.SEGMENTATION)
    training_data.create()

    validation_data = ImageData(file_paths=file_paths.validation_paths,
                                batch_size=config.TrainingParams.batch_size_val,
                                buffer_size=config.TrainingParams.buffer_size_val,
                                shuffle=config.DataParams.shuffle,
                                do_pre_processing=False,
                                mode=config.DataModes.VALIDATION,
                                train_mode=config.TrainingModes.SEGMENTATION)
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
                    freeze_layers=freeze)

    opt_args = None
    if config.TrainingParams.optimizer == config.Optimizer.MOMENTUM:
        opt_args = config.TrainingParams.momentum_args
    elif config.TrainingParams.optimizer == config.Optimizer.ADAGRAD:
        opt_args = config.TrainingParams.adagrad_args
    elif config.TrainingParams.optimizer == config.Optimizer.ADAM:
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
                         caffemodel_path=caffemodel_path)