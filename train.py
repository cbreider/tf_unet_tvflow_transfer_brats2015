"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019
"""


from src.utils.path_utils import DataPaths
from src.utils.split_utilities import TrainingDataset
from src.tf_data_pipeline_wrapper import ImageData
from src.tf_convnet import tf_convnet
import src.utils.gpu_selector as cuda_selector
import src.trainer as trainer
import argparse
import configuration as config
import tensorflow as tf
import logging
import os
from src.utils.enum_params import TrainingModes, DataModes, Optimizer, RestoreMode


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        help="Mode of training session. 1=TV regression training, 2=TV clustering training, "
                             "3=BRATS Segmentation",
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
    parser.add_argument("--caffemodel_path",
                        help="Path of a pretrained caffe model (hfd5). If given it will load this model",
                        type=str, default=None)
    parser.add_argument('--cuda_device', type=int, default=-1,
                        help='Number of cuda device to use (optional)')
    parser.add_argument("--restore_mode",
                        help="Mode of restoring session. 1=Complete Session, 2=Without Out Layer, 3=Complete Net. "
                             "Only used if restore_path is given",
                        type=int, default=1)

    args = parser.parse_args()
    create_new_training_split = False
    create_summaries = True
    restore_path = None
    caffemodel_path = None
    train_mode = TrainingModes(args.mode)
    restore_mode = RestoreMode(args.restore_mode)

    if args.create_new_split:
        create_new_training_split = True
    if args.do_not_create_summaries:
        create_summaries = False
    if args.restore_path is not None:
        restore_path = args.restore_path
    if args.caffemodel_path is not None:
        caffemodel_path = args.caffemodel_path
    if args.cuda_device >= 0:
        cuda_selector.set_cuda_gpu(args.cuda_device)

    # tf.enable_eager_execution()
    tf.reset_default_graph()

    data_paths = DataPaths(data_path="default", mode=train_mode.name)
    data_paths.load_data_paths(mkdirs=True, restore_dir=restore_path if (restore_mode == RestoreMode.COMPLETE_SESSION
                                                                         or restore_mode == RestoreMode.COMPLETE_NET)
                                                                    else None)

    outF = open(os.path.join(data_paths.tf_out_path, "args.txt"), "w")
    outF.write("create_new_split: {}".format(create_new_training_split))
    outF.write("\n")
    outF.write("restore_path: {}".format(restore_path))
    outF.write("\n")
    outF.write("caffemodel_path: {}".format(caffemodel_path))
    outF.write("\n")
    outF.write("restore_mode: {}".format(restore_mode))
    outF.write("\n")
    outF.close()

    file_paths = TrainingDataset(paths=data_paths, mode=train_mode, data_config=config.DataParams,
                                 new_split=create_new_training_split)

    training_data = ImageData(data=file_paths.train_paths,
                              mode=DataModes.TRAINING,
                              train_mode=train_mode,
                              data_config=config.DataParams)

    validation_data = ImageData(data=file_paths.validation_paths,
                                mode=DataModes.VALIDATION,
                                train_mode=train_mode,
                                data_config=config.DataParams)

    training_data.create()
    validation_data.create()

    net = tf_convnet.ConvNetModel(convnet_config=config.ConvNetParams, create_summaries=create_summaries)

    opt_args = None
    if config.TrainingParams.optimizer == Optimizer.MOMENTUM:
        opt_args = config.TrainingParams.momentum_args
    elif config.TrainingParams.optimizer == Optimizer.ADAGRAD:
        opt_args = config.TrainingParams.adagrad_args
    elif config.TrainingParams.optimizer == Optimizer.ADAM:
        opt_args = config.TrainingParams.adam_args

    trainer = trainer.Trainer(net=net, data_provider_train=training_data, data_provider_val=validation_data,
                              out_path=data_paths.tf_out_path, train_config=config.TrainingParams,
                              restore_path=restore_path, caffemodel_path=caffemodel_path,
                              restore_mode=restore_mode, mode=train_mode)

    path = trainer.train()


