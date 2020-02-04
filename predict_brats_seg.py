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
from src.tf_convnet.tf_convnet import ConvNetModel
import src.utils.data_utils as dutils
import src.utils.gpu_selector as cuda_selector
import argparse
import os
import configuration as config
import tensorflow as tf
import logging
import src.utils.logger as log
from random import *
from src.utils.enum_params import TrainingModes, DataModes, Optimizer, RestoreMode
from src.tf_data_pipeline_wrapper import ImageData

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
                                     mode=TrainingModes.SEGMENTATION,
                                     data_config=config.DataParams,
                                     load_test_paths_only=True,
                                     new_split=False)

    out_path = os.path.join(model_path, "predictions")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    data = ImageData(data=file_paths.test_paths, mode=DataModes.VALIDATION, train_mode=TrainingModes.SEGMENTATION,
                     data_config=config.DataParams)

    data.create()

    net = ConvNetModel(convnet_config=config.ConvNetParams, create_summaries=True)

    idx = 0
    accuracy = 0
    error = 0
    dice = 0
    cross_entropy = 0

    with tf.Session() as sess:
        logging.info("Running Tests on {} samples".format(len(file_paths.test_paths)))
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            net.restore(sess, ckpt.model_checkpoint_path, restore_mode=RestoreMode.COMPLETE_SESSION)
        sess.run(data.init_op)
        if not use_Brats_Testing:
            for i in range(int(len(file_paths.test_paths)/config.TrainingParams.batch_size_val)):
                batch_x, batch_y, __ = sess.run(data.next_batch)

                prediction, ce, dc, err, acc, fmaps = sess.run([net.predicter,
                                                         net.cross_entropy,
                                                         net.dice,
                                                         net.error,
                                                         net.accuracy,
                                                         net.last_feature_map],
                                                        feed_dict={net.x: batch_x,
                                                                   net.y: batch_y,
                                                                   net.keep_prob: 1.})
                accuracy += acc
                error += err
                dice += dc
                cross_entropy += ce
                fname = "{}.jpg".format(i)
                st = randint(1, 100)
                if st == 100 or save_all_predictions:
                    img = dutils.combine_img_prediction(batch_x, batch_y, prediction, mode=0)
                    dutils.save_image(img, os.path.join(out_path, fname))
                if save_fmaps:
                    size = [8, 8]
                    map_s = [fmaps.shape[1], fmaps.shape[2]]
                    fmaps = dutils.revert_zero_centering(fmaps)
                    for m in range(fmaps.shape[0]):
                        fmap = fmaps[m]
                        im = fmap.reshape(map_s[0], map_s[0],
                                          size[0], size[1]).transpose(2, 0,
                                                                      3, 1).reshape(size[0]*map_s[0],
                                                                                    size[1]*map_s[1])
                        # histogram normalization
                        im = dutils.image_histogram_equalization(im)[0]
                        dutils.save_image(im, os.path.join(out_path, "{}_{}.jpg".format(i, m)))

                logging.info(
                    "Test {} of {}  finished".format(idx,
                                                     int(len(file_paths.test_paths)/config.TrainingParams.batch_size_val)))

                idx += 1

    accuracy /= idx
    error /= idx
    dice /= idx
    cross_entropy /= idx

    outF = open(os.path.join(model_path, "results.txt"), "w")

    outF.write("ERROR: {}".format(error))
    outF.write("\n")
    outF.write("ACCURACY: {}".format(accuracy))
    outF.write("\n")
    outF.write("CROSS ENTROPY: {}".format(cross_entropy))
    outF.write("\n")
    outF.write("DICE: {}".format(dice))
    outF.write("\n")
    outF.close()

    logging.info("Test successfully evaluated:")
    logging.info("ERROR: {}".format(error))
    logging.info("ACCURACY: {}".format(accuracy))
    logging.info("CROSS ENTROPY: {}".format(cross_entropy))
    logging.info("DICE: {}".format(dice))









