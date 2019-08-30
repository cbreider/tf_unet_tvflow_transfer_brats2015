from src.utils.path_utils import DataPaths
from src.utils.split_utilities import TrainingDataset
from src.tf_unet import unet
import src.utils.data_utils as dutils
import src.utils.gpu_selector as cuda_selector
import argparse
import os
import configuration as config
import tensorflow as tf
import logging
import numpy as np
from random import *
from src.utils.enum_params import TrainingModes, DataModes, Optimizer, RestoreMode

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


if __name__ == "__main__":

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

    args = parser.parse_args()

    model_path = None
    use_Brats_Testing = False
    save_all_predictions = False
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
    if args.cuda_device >= 0:
        cuda_selector.set_cuda_gpu(args.cuda_device)

    # tf.enable_eager_execution()
    tf.reset_default_graph()
    out_path=model_path
    data_paths = DataPaths(data_path="default", mode="SEGMENTATION")
    data_paths.load_data_paths()
    file_paths = None
    if not use_Brats_Testing:
        file_paths = TrainingDataset(paths=data_paths,
                                     mode=TrainingModes.SEGMENTATION,
                                     load_test_paths_only=True)

    out_path = os.path.join(model_path, "predictions")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    net = unet.Unet(n_channels=config.DataParams.nr_of_channels,
                    n_class=config.DataParams.nr_of_classes_seg_mode,
                    cost_function=config.ConvNetParams.cost_function,
                    summaries=False,
                    class_weights=config.ConvNetParams.class_weights,
                    regularizer=config.ConvNetParams.regularizer,
                    n_layers=config.ConvNetParams.num_layers,
                    features_root=config.ConvNetParams.feat_root,
                    filter_size=config.ConvNetParams.filter_size,
                    pool_size=config.ConvNetParams.pool_size,
                    use_padding=config.ConvNetParams.padding,
                    batch_norm=config.ConvNetParams.batch_normalization,
                    add_residual_layer=config.ConvNetParams.add_residual_layer,
                    use_scale_image_as_gt=config.DataParams.use_scale_image_as_gt,
                    act_func_out=config.ConvNetParams.activation_func_out)

    with tf.Session as sess:
        net.restore(sess=sess, model_path=model_path, RestoreMode=RestoreMode.COMPLETE_SESSION)
        if not use_Brats_Testing:
            idx = 0
            accuracy = 0
            error = 0
            dice = 0
            cross_entropy = 0
            predictions = []

            for gt, input in file_paths.test_paths.items():
                predictions = []
                name = input.replace(data_paths.data_dir + "/", "" )
                gt = dutils.load_3d_volume_as_array(gt)
                input_scan = dutils.load_3d_volume_as_array(input)
                input_scan = dutils.intensity_normalize_one_volume(input_scan, norm_std=config.DataParams.normailze_std)
                input_scan = input_scan / np.max(input_scan)

                for i in range(gt.size[2]):
                    prediction, ce, dc, err, acc = sess.run([net.predicter,
                                                         net.cross_entropy,
                                                         net.dice, net.error,
                                                         net.accuracy],
                                                            feed_dict={net.x: input_scan[i],
                                                                       net.y: gt[i],
                                                                       net.keep_prob: 1.})
                    accuracy += acc
                    error += err
                    dice += dc
                    cross_entropy += ce
                    predictions[i] = prediction
                    fname = "{}_{}.jpg".format(name, i)
                    st = randint(1, 100)
                    if st == 100:
                        img = dutils.combine_img_prediction(input_scan, gt, prediction, mode=0)
                        dutils.save_image(img, os.path.join(out_path, name))

                    idx += 1

            if save_all_predictions:
                dutils.save_array_as_nifty_volume(predictions, os.path.join(model_path, name + ".mha"))


            accuracy /= idx
            error /= idx
            dice /= idx
            ce /= idx

            outF = open(os.path.join(model_path, "results.txt", "w"))

            outF.write("ERROR: {}".format(error))
            outF.write("\n")
            outF.write("ACCURACY: {}".format(accuracy))
            outF.write("\n")
            outF.write("CROSS ENTROPY: {}".format(cross_entropy))
            outF.write("\n")
            outF.write("DICE: {}".format(dice))
            outF.write("\n")
            outF.close()

            print("Test successfully evaluated:")
            print("ERROR: {}".format(error))
            print("ACCURACY: {}".format(accuracy))
            print("CROSS ENTROPY: {}".format(cross_entropy))
            print("DICE: {}".format(dice))









