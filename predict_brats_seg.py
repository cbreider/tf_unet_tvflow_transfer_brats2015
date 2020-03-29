"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""

from src.utils.path_utils import DataPaths
from src.utils.split_utilities import TestFilePaths
from src.tf_convnet.tf_convnet import ConvNetModel
import src.utils.gpu_selector as cuda_selector
import argparse
import os
import configuration as config
import tensorflow as tf
import logging
import  src.utils.io_utils as ioutil
import src.utils.logger as log
from src.validator import Validator
from src.utils.enum_params import TrainingModes, DataModes, Optimizer, RestoreMode
from src.tf_data_pipeline_wrapper import ImageData
import numpy as np
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


if __name__ == "__main__":

    logging.basicConfig(filename='log_test.log', level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path",
                        help="Path of a stored model. If given it will load this model",
                        type=str, default=None, required=True)
    parser.add_argument("--name",
                        help="Name of the files. VSD.[name].patientID.mha",
                        type=str, default=None, required=True)
    parser.add_argument("--use_brats_test_set",
                        help="Loads Scans from Brats test set instead of own test set from BRATS train samples",
                        action='store_true')
    parser.add_argument('--cuda_device', type=int, default=-1,
                        help='Number of cuda device to use (optional)')
    parser.add_argument('--save_pngs', action='store_true',
                        help='saves the feature maps of the last CNN layer')
    parser.add_argument('--take_fold_nr', type=int, default=0,
                        help='use_fold')

    args = parser.parse_args()

    model_path = None
    name = "Test"
    use_Brats_Testing = False
    save_all_predictions = False
    save_pngs = False
    batch_size = 1
    if args.model_path is not None:
        model_path = args.model_path
    else:
        raise ValueError()
    if args.use_brats_test_set:
        use_Brats_Testing = True
        print("TODO")
        exit(1)
    if args.save_pngs:
        save_pngs = True
    if args.cuda_device >= 0:
        cuda_selector.set_cuda_gpu(args.cuda_device)
    if args.name:
        name = args.name
    fold_nr = args.take_fold_nr

    # tf.enable_eager_execution()
    tf.reset_default_graph()
    out_path=model_path
    data_paths = DataPaths(data_path="default", mode="SEGMENTATION_TEST",
                           tumor_mode=config.DataParams.segmentation_mask.name)
    data_paths.load_data_paths(mkdirs=False, restore_dir=model_path)

    log.init_logger(type="test", path=data_paths.tf_out_path)
    logging.info("Allocating '{}'".format(data_paths.tf_out_path))
    if use_Brats_Testing:
        patient_paths = [os.path.join(data_paths.raw_test_dir, path) for path in os.listdir(data_paths.raw_test_dir)]
    else:
        split_file = os.path.join(data_paths.split_path, "split.json")
        if not os.path.exists(split_file):
            raise ValueError('{0:} file not found'.format(split_file))
        file = open(split_file, 'r')
        data = file.read()
        split = json.loads(data)
        patient_paths = split["testing"]

    file_paths = TestFilePaths(paths=data_paths, patient_paths=patient_paths, config=config.DataParams)

    out_path_base = os.path.join(model_path, "predictions")
    out_path_mha = os.path.join(out_path_base, "mha")
    out_path_png = os.path.join(out_path_base, "png")

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(out_path_mha):
        os.makedirs(out_path_mha)
    if not os.path.exists(out_path_png) and save_pngs:
        os.makedirs(out_path_png)

    data_iter = ImageData(data=file_paths.test_paths, mode=DataModes.TESTING,
                          train_mode=TrainingModes.BRATS_SEGMENTATION, data_config=config.DataParams)
    data_iter.create()

    net = ConvNetModel(convnet_config=config.ConvNetParams, mode=TrainingModes.BRATS_SEGMENTATION,
                       create_summaries=True)
    data = [[], []]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            net.restore(sess, ckpt.model_checkpoint_path, restore_mode=RestoreMode.COMPLETE_SESSION)
        sess.run(data_iter.init_op)
        for i in range(int(data_iter.size / batch_size)):

            test_x, pat_id = sess.run(data_iter.next_batch)
            pat_id = id[0].decode("utf-8")

            prediction = sess.run(net.predicter,
                                  feed_dict={net.x: test_x,
                                             net.keep_prob_conv1: 1.0,
                                             net.keep_prob_conv2: 1.0,
                                             net.keep_prob_pool: 1.0,
                                             net.keep_prob_tconv: 1.0,
                                             net.keep_prob_concat: 1.0})

            data[0].append(np.squeeze(np.array(test_x), axis=0))
            data[1].append(np.squeeze(np.array(prediction), axis=0))

            shape = prediction.shape

            if len(data[1]) == 155:

                file_name = "VSD.{}.{}".format(name, pat_id)
                pred_slice = np.argmax(np.array(data[1]), axis=3).astype(np.int16)
                mha_path = png_path = os.path.join(out_path_mha, "{}.mha".format(file_name))
                ioutil.save_scan_as_mha(pred_slice, mha_path)

                if save_pngs:
                    png_path = "{}.png".format(file_name)
                    Validator.store_prediction(file_name, mode=TrainingModes.BRATS_SEGMENTATION, path=out_path_png,
                                               batch_x=np.array(data[0]), batch_y=np.array(data[0]),
                                               batch_tv=np.array(data[0]), prediction=np.array(data[1]),
                                               gt_is_one_hot=True)

                data = [[], []]

            ioutil.progress(i, int(data_iter.size / batch_size))
            
        if len(data[1]) > 0:
            raise ValueError("Something seems to be wrong with number of scans")








