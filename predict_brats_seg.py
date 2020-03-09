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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


if __name__ == "__main__":

    logging.basicConfig(filename='log_test.log', level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path",
                        help="Path of a stored model. If given it will load this model",
                        type=str, default=None, required=True)
    parser.add_argument("--name",
                        help="Name of the files. VSD.[name].patientID.mha",
                        type=str, default=None)
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
    if args.use_Brats_Testing:
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
    file_paths = None

    log.init_logger(type="test", path=data_paths.tf_out_path)
    logging.info("Allocating '{}'".format(data_paths.tf_out_path))

    if not use_Brats_Testing:
        file_paths = TestFilePaths(paths=data_paths,
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
        """
        for i in range(int(data.size / batch_size)):

            test_x, id = sess.run(data.next_batch)

            prediction = sess.run(net.predicter,
                                  feed_dict={net.x: test_x,
                                             net.keep_prob: 1.})

            data[0].append(np.squeeze(np.array(test_x), axis=0))
            data[3].append(np.squeeze(np.array(prediction), axis=0))
            shape = prediction.shape

            if len(data[1]) == 155:

                file_name = "VSD.{}.{}".format(name, id)
                pred_slice = np.argmax(np.array(data[3]), axis=3).astype(float)

                #if save_pngs:
                    #Validator.store_prediction(file_name, self._mode, self._output_path,
                    #                      np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3]),
                    #                      gt_is_one_hot=False if self._conv_net.cost == Cost.MSE else True)


                data = [[], [], [], [], []]

            ioutil.progress(i, int(data.size / batch_size))
            
            """
        if len(data[1]) > 0:
            raise ValueError("Somthing seems to be wrong with number of scans")








