"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""

from src.utils.path_utils import DataPaths
from src.utils.split_utilities import TrainingDataset
from src.tf_convnet.tf_convnet import ConvNetModel
import src.utils.gpu_selector as cuda_selector
import argparse
import os
import configuration as config
import tensorflow as tf
import logging
import src.utils.logger as log
from src.utils.enum_params import TrainingModes, DataModes, Optimizer, RestoreMode
from src.tf_data_pipeline_wrapper import ImageData
import src.validator as vali


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


if __name__ == "__main__":

    logging.basicConfig(filename='log_test.log', level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path",
                        help="Path of a stored model. If given it will load this model",
                        type=str, default=None)
    parser.add_argument("--use_Brats_Testing",
                        help="Loads a test set fom split path instead of Brats testing",
                        action='store_true')
    parser.add_argument('--cuda_device', type=int, default=-1,
                        help='Number of cuda device to use (optional)')
    parser.add_argument('--save_all_predictions', action='store_true',
                        help='save all predictions as mha to model_path')
    parser.add_argument('--save_feature_maps', action='store_true',
                        help='saves the feature maps of the last CNN layer')
    parser.add_argument('--take_fold_nr', type=int, default=0,
                        help='use_fold')

    args = parser.parse_args()

    model_path = None
    use_Brats_Testing = False
    save_all_predictions = False
    save_fmaps = False
    if args.model_path is not None:
        model_path = args.model_path
    else:
        raise ValueError()
    if args.use_Brats_Testing:
        use_Brats_Testing = True
        print("TODO")
        exit(1)
    if args.save_all_predictions:
        save_all_predictions = True
    if args.save_feature_maps:
        save_fmaps = True
    if args.cuda_device >= 0:
        cuda_selector.set_cuda_gpu(args.cuda_device)
    fold_nr = args.take_fold_nr

    # tf.enable_eager_execution()
    tf.reset_default_graph()
    out_path=model_path
    data_paths = DataPaths(data_path="default", mode="SEGMENTATION_TEST",
                           tumor_mode=config.DataParams.segmentation_mask.name)
    data_paths.load_data_paths(mkdirs=False, restore_dir=model_path)
    file_paths = None

    log.init_logger(type="test", path=data_paths.tf_out_path)
    logging.info("Allocating '{}'".format(data_paths.tf_out_path))

    if not use_Brats_Testing:
        file_paths = TrainingDataset(paths=data_paths,
                                     mode=TrainingModes.BRATS_SEGMENTATION,
                                     data_config=config.DataParams,
                                     load_test_paths_only=True,
                                     new_split=False,
                                     is_five_fold=True if fold_nr > 0 else False,
                                     five_fold_idx=fold_nr)

    out_path = os.path.join(model_path, "predictions")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    data = ImageData(data=file_paths.test_paths, mode=DataModes.VALIDATION, train_mode=TrainingModes.BRATS_SEGMENTATION,
                     data_config=config.DataParams)

    data.create()

    net = ConvNetModel(convnet_config=config.ConvNetParams, mode=TrainingModes.BRATS_SEGMENTATION,
                       create_summaries=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            net.restore(sess, ckpt.model_checkpoint_path, restore_mode=RestoreMode.COMPLETE_SESSION)
        sess.run(data.init_op)
        if not use_Brats_Testing:
            vali.run_test(sess, net, data_provider_test=data, mode=TrainingModes.BRATS_SEGMENTATION, nr=fold_nr,
                          out_path="")







