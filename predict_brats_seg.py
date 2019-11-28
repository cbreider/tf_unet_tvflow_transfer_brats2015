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
from src.tf_data_pipeline_wrapper import ImageData

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

    data = ImageData(data=file_paths.test_paths,
                                batch_size=config.TrainingParams.batch_size_val,
                                buffer_size=config.TrainingParams.buffer_size_val,
                                shuffle=False,
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
    data.create()

    net = unet.Unet(n_channels=config.DataParams.nr_of_channels,
                    n_class=config.DataParams.nr_of_classes_seg_mode,
                    cost_function=config.ConvNetParams.cost_function,
                    summaries=True,
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

    idx = 0
    accuracy = 0
    error = 0
    dice = 0
    cross_entropy = 0

    with tf.Session() as sess:
        print("Running Tests on {} samples".format(len(file_paths.test_paths)))
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            net.restore(sess, ckpt.model_checkpoint_path, restore_mode=RestoreMode.ONLY_BASE_NET)
        sess.run(data.init_op)
        if not use_Brats_Testing:
            for i in range(int(len(file_paths.test_paths)/config.TrainingParams.batch_size_val)):
                batch_x, batch_y, __ = sess.run(data.next_batch)

                prediction, ce, dc, err, acc, map= sess.run([net.predicter,
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

                map = dutils.revert_zero_centering(map)
                for m in range(map.shape[0]):
                    x = map[0]
                    x = np.reshape(x, (x.shape[1]*x.shape[3], x.shape[2]*x.shape[3]))
                    dutils.save_image(x, os.path.join(out_path, "{}_{}.jpg".format(i, m)))

                print("Test {} of {}  finished".format(idx, int(len(file_paths.test_paths)/config.TrainingParams.batch_size_val)))

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

    print("Test successfully evaluated:")
    print("ERROR: {}".format(error))
    print("ACCURACY: {}".format(accuracy))
    print("CROSS ENTROPY: {}".format(cross_entropy))
    print("DICE: {}".format(dice))









