"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""

import os
import src.utils.data_utils as dutil
import src.utils.io_utils as ioutil
import numpy as np
from src.utils.enum_params import Cost, Optimizer, RestoreMode, TrainingModes


class Validator(object):

    def __init__(self, tf_session, net, data_provider, out_path, mode, nr=1, mini_validation=False,
                 store_feature_maps=True, store_predictions=True):

        """
        Trains a convnet instance

        :param net: the unet instance to train
        :param data_provider_train: callable returning training and verification data
        :param data_provider_val: callable returning training and verification data
        :param out_path: path to store summaries and checkpoints
        :param train_config: parameters for  training # type: TrainingParams
        :param restore_path: (optional) checkpoint path to restore a tf checkpoint
        :param caffemodel_path: (optional) path to a unet caffemodel if you want to restore one
        :param restore_mode: (optional) mode how to restore a tf checkpoint
        """
        self._tf_session = tf_session
        self._conv_net = net
        self._data_provider = data_provider
        self._output_path = out_path
        self._mini = mini_validation
        self._nr = nr
        self._store_fmaps = store_feature_maps
        self._store_predictions = store_predictions
        self._mode = mode # type: TrainingModes
        self.fmap_size = [8, 8]
        self._batch_size = 1

    def run_validation(self):
        mini_size = 5
        vals = []
        dice_per_volume = []
        data = [[], [], [], [], []]  # x, y, tv, pred, feature maps
        shape = []
        itr = 0
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)
        self._tf_session.run(self._data_provider.init_op)

        set_size = int(self._data_provider.size / self._batch_size)
        ioutil.progress(0, set_size if not self._mini else (mini_size * 155))

        for i in range(int(self._data_provider.size / self._batch_size)):
            test_x, test_y, test_tv = self._tf_session.run(self._data_provider.next_batch)
            loss, acc, err, err_r, prediction, dice, ce, iou, feature = self._tf_session.run(
                [self._conv_net.cost, self._conv_net.accuracy, self._conv_net.error, self._conv_net.error_rate,
                 self._conv_net.predicter, self._conv_net.dice, self._conv_net.cross_entropy, self._conv_net.iou_coe,
                 self._conv_net.last_feature_map],
                feed_dict={self._conv_net.x: test_x,
                           self._conv_net.y: test_y,
                           self._conv_net.keep_prob: 1.})

            vals.append([loss, ce, dice, err, err_r, acc, iou])
            data[0].append(test_x)
            data[1].append(test_y)
            data[2].append(test_tv)
            data[3].append(prediction)
            data[4].append(feature)
            shape = prediction.shape

            if len(data[1]) == 155:
                dice_per_volume.append(dutil.get_hard_dice_score(np.array(data[1]), np.array(data[:][3])))
                if self._store_predictions:
                    self.store_prediction("{}_{}".format(self._nr, itr), self._mode, self._output_path,
                                          np.squeeze(np.array(data[0]), axis=1), np.squeeze(np.array(data[1]), axis=1),
                                          np.squeeze(np.array(data[2]), axis=1), np.squeeze(np.array(data[3]), axis=1),
                                          gt_is_one_hot=False if self._conv_net.cost == Cost.MSE else True)

                # safe one feature_map
                if self._store_fmaps:
                    fmap = dutil.revert_zero_centering(np.squeeze(np.array(data[4][75]), axis=0))
                    map_s = [fmap.shape[0], fmap.shape[1]]
                    im = fmap.reshape(map_s[0], map_s[0], self.fmap_size[0], self.fmap_size[1]
                                      ).transpose(2, 0, 3, 1
                                                  ).reshape(self.fmap_size[0] * map_s[0], self.fmap_size[1] * map_s[1])
                    # histogram normalization
                    # im = util.image_histogram_equalization(im)[0]
                    ioutil.save_image(im, os.path.join(self._output_path, "{}_{}_fmap.jpg".format(self._nr, itr)))

                data = [[], [], [], [], []]
                itr += 1
                if self._mini and itr == mini_size:
                    break

            ioutil.progress(i, set_size if not self._mini else (mini_size * 155))

        if len(data[1]) > 0:
            if self._store_predictions:
                self.store_prediction("{}_{}".format(self._nr, itr), self._mode, self._output_path,
                                      np.squeeze(np.array(data[0]), axis=1), np.squeeze(np.array(data[1]), axis=1),
                                      np.squeeze(np.array(data[2]), axis=1), np.squeeze(np.array(data[3]), axis=1),
                                      gt_is_one_hot=False if self._conv_net.cost == Cost.MSE else True)
        val_scores = np.mean(np.array(vals), axis=0)
        dp = np.mean(np.array(dice_per_volume))

        return shape, np.append(val_scores, dp)

    @staticmethod
    def store_prediction(name, mode, path, batch_x, batch_y, batch_tv, prediction, gt_is_one_hot=True):
        if mode == TrainingModes.TVFLOW_SEGMENTATION:
            img = dutil.combine_img_prediction_tvclustering(data=batch_x, gt=batch_y, tv=batch_tv, pred=prediction)
        else:
            img = dutil.combine_img_prediction(batch_x, batch_y, prediction,
                                               mode=0 if gt_is_one_hot else 1)
        ioutil.save_image(img, "%s/%s.jpg" % (path, name))