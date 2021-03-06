"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""

import os
import src.utilities.np_data_process_utils as dutil
import src.utilities.io_utils as ioutil
import numpy as np
from src.utilities.enum_params import Cost, TrainingModes, Scores
import logging
import collections


def run_test(sess, net, data_provider_test, mode, nr, out_path):
    logging.info("Running evaluation on {} test images ....".format(data_provider_test.size))
    test_out_path = os.path.join(out_path, "Test_{}".format(nr))
    pred_shape, validation_results = Validator(sess, net, data_provider_test,
                                               test_out_path, mode=mode, mini_validation=False,
                                               nr=nr, store_feature_maps=False,
                                               store_predictions=False, write_per_patient_result=True).run_validation()
    l_string = "TEST RESULTS:"
    for k, v in validation_results.items():
        if v != -1.:
            l_string = "{} {}= {:.6f},".format(l_string, k.value, v)
    l_string = l_string[:-1]
    logging.info(l_string)

    if not os.path.exists(test_out_path):
        os.makedirs(test_out_path)

    outF = open(os.path.join(test_out_path, "results.txt"), "w")
    for k, v in validation_results.items():
        if v != -1.:
            outF.write("{}: {:6f}".format(k, v))
            outF.write("\n")
    outF.close()


class Validator(object):

    def __init__(self, tf_session, net, data_provider, out_path, mode, nr=1, mini_validation=False,
                 store_feature_maps=True, store_predictions=True, write_per_patient_result=False):

        """
        Trains a convnet instance

        :param tf_session: initialized tensorflow session
        :param net: the unet instance to train
        :param data_provider: callable returning verification data
        :param mode: mode of training (BraTS segmentation, tv regression etc.)
        :param out_path: path to store summaries and checkpoints
        :param mini_validation: bool indicator if only use a small subset of overall validation data.
        :param store_feature_maps: store feature map of last layer for each prediction as png
        :param store_predictions: store predictions as png
        :param write_per_patient_result: write dice score per patient to csv file
        :param nr: Nr of the epcoh
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
        self.write_per_patient_result = write_per_patient_result

    def run_validation(self):
        mini_size = 5
        vals = []
        dices_per_volume = []
        precison_per_volume = []
        sensitivity_per_volume = []
        specificity_per_volume = []
        data = [[], [], [], [], []]  # x, y, tv, pred, feature maps
        shape = []
        itr = 0
        dice_complete = -1.
        dice_core = -1.
        dice_enhancing = -1.
        dice_overall = -1.
        precision_core = -1.
        sensitivity_core = -1.
        specificity_core = -1.
        precision_complete = -1.
        sensitivity_complete = -1.
        specificity_complete = -1.
        precision_enhancing = -1.
        sensitivity_enhancing = -1.
        specificity_enhancing = -1.
        precision = -1.
        sensitivity = -1.
        specificity = -1.
        pat_ids = []
        if not os.path.exists(self._output_path) and(self._store_predictions or self._store_fmaps):
            os.makedirs(self._output_path)
        self._tf_session.run(self._data_provider.init_op)

        set_size = int(self._data_provider.size / self._batch_size)
        ioutil.progress(0, set_size if not self._mini else (mini_size * 155))

        for i in range(int(self._data_provider.size / self._batch_size)):
            test_x, test_y, test_tv, paths = self._tf_session.run(self._data_provider.next_batch)
            [loss, acc, err, prediction, dice, ce, iou, feature, d_complete, d_core, d_enhancing] = self._tf_session.run(
                [self._conv_net.cost, self._conv_net.accuracy, self._conv_net.error,
                 self._conv_net.predicter, self._conv_net.dice, self._conv_net.cross_entropy, self._conv_net.iou_coe,
                 self._conv_net.last_feature_map, self._conv_net.dice_complete, self._conv_net.dice_core,
                 self._conv_net.dice_enhancing],
                feed_dict={self._conv_net.x: test_x,
                           self._conv_net.y: test_y,
                           self._conv_net.keep_prob_conv1: 1.0,
                           self._conv_net.keep_prob_conv2: 1.0,
                           self._conv_net.keep_prob_pool: 1.0,
                           self._conv_net.keep_prob_tconv: 1.0,
                           self._conv_net.keep_prob_concat: 1.0})

            vals.append([loss, ce, err, acc, iou, dice, d_complete, d_core, d_enhancing])
            data[0].append(np.squeeze(np.array(test_x), axis=0))
            data[1].append(np.squeeze(np.array(test_y), axis=0))
            data[2].append(np.squeeze(np.array(test_tv), axis=0))
            data[3].append(np.squeeze(np.array(prediction), axis=0))
            data[4].append(feature)
            shape = prediction.shape

            if len(data[1]) == 155:
                pat_ids.append(
                    [str(f) for f in paths[0] if "mr_flair" in str(f).lower()][0].split(".")[-2].split("_")[0])
                if (self._mode == TrainingModes.BRATS_SEGMENTATION or self._mode == TrainingModes.BRATS_SEGMENTATION_TV_PSEUDO_PATIENT) and np.shape(data[1])[3] > 1:
                    pred_slice = np.argmax(np.array(data[3]), axis=3).astype(float)
                    y_slice = np.argmax(np.array(data[1]), axis=3).astype(float)
                    pred_complete = np.greater(pred_slice, 0.).astype(float)
                    pred_core =np.logical_or(np.logical_or(np.equal(pred_slice, 1.),
                                                            np.equal(pred_slice, 3.)),
                                                  np.equal(pred_slice, 4.)).astype(float)
                    pred_enhancing = np.equal(pred_slice, 4.).astype(float)
                    y_complete = np.greater(y_slice, 0.).astype(float)
                    y_core = np.logical_or(np.logical_or(np.equal(y_slice, 1.), np.equal(y_slice, 3.)),
                                               np.equal(y_slice, 4.)).astype(float)
                    y_enhancing = np.equal(y_slice, 4.).astype(float)
                    dice_complete = dutil.get_hard_dice_score(pred=pred_complete, gt=y_complete, eps=1e-5, axis=(0,1,2))
                    dice_core = dutil.get_hard_dice_score(pred=pred_core, gt=y_core, eps=1e-5, axis=(0,1,2))
                    dice_enhancing = dutil.get_hard_dice_score(pred=pred_enhancing, gt=y_enhancing, eps=1e-5, axis=(0,1,2))
                    dice_overall = (dice_complete + dice_core + dice_enhancing) / 3.0
                    [precision_core, sensitivity_core, specificity_core] = dutil.get_confusion_metrices(pred_core, y_core)
                    [precision_complete, sensitivity_complete,
                     specificity_complete] = dutil.get_confusion_metrices(pred_complete, y_complete)
                    [precision_enhancing, sensitivity_enhancing,
                     specificity_enhancing] = dutil.get_confusion_metrices(pred_enhancing, y_enhancing)
                elif self._conv_net.cost_function != Cost.MSE:
                    dice_overall = dutil.get_hard_dice_score(np.array(data[3]), np.array(data[1]), axis=(0,1,2,3))
                    [precision, sensitivity, specificity] = dutil.get_confusion_metrices(np.array(data[1]),
                                                                                         np.array(data[3]))

                dices_per_volume.append([dice_overall, dice_complete, dice_core, dice_enhancing])
                precison_per_volume.append([precision, precision_complete, precision_core, precision_enhancing])
                sensitivity_per_volume.append([sensitivity, sensitivity_complete, sensitivity_core,
                                               sensitivity_enhancing])
                specificity_per_volume.append([specificity, specificity_complete, specificity_core,
                                               specificity_enhancing])

                if self._store_predictions:
                    self.store_prediction("{}_{}".format(self._nr, itr), self._mode, self._output_path,
                                          np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3]),
                                          gt_is_one_hot=0 if self._conv_net.cost_function == Cost.MSE else 1)

                # safe one feature_map
                if self._store_fmaps:
                    fmap = np.squeeze(np.array(data[4][75]), axis=0)
                    map_s = [fmap.shape[0], fmap.shape[1]]
                    im = fmap.reshape(map_s[0], map_s[0], self.fmap_size[0], self.fmap_size[1]
                                      ).transpose(2, 0, 3, 1
                                                  ).reshape(self.fmap_size[0] * map_s[0], self.fmap_size[1] * map_s[1])
                    im -= im.min()
                    im /= im.max()
                    im *= 255.0
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
                                      np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3]),
                                      gt_is_one_hot=0 if self._conv_net.cost_function == Cost.MSE else 1)
        sbatch = np.mean(np.array(vals), axis=0)

        if self.write_per_patient_result:
            out_str = "DSC;DSC_complete;DSC_Core;DSC_enhancing;PatientID\n"
            idx = 0
            for score in dices_per_volume:
                out_str = "{}{};{};{};{};{}\n".format(out_str, score[0], score[1], score[2], score[3], pat_ids[idx])
                idx += 1

            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)
            
            outF = open(os.path.join(self._output_path, "results_per_patient.csv"), "w")
            outF.write(out_str)
            outF.write("\n")
            outF.close()

        d_per_patient = np.mean(np.array(dices_per_volume), axis=0)
        p_per_patient = np.mean(np.array(precison_per_volume), axis=0)
        sen_per_patient = np.mean(np.array(sensitivity_per_volume), axis=0)
        spec_per_patient = np.mean(np.array(specificity_per_volume), axis=0)

        scores = collections.OrderedDict()

        scores[Scores.LOSS] = sbatch[0]
        scores[Scores.CE] = sbatch[1]
        scores[Scores.ERROR] = sbatch[2]
        scores[Scores.ACC] = sbatch[3]
        scores[Scores.IOU] = sbatch[4]
        scores[Scores.DSC] = sbatch[5]
        scores[Scores.DSC_COMP] = sbatch[6]
        scores[Scores.DSC_CORE] = sbatch[7]
        scores[Scores.DSC_EN] = sbatch[8]

        scores[Scores.DSCP] = d_per_patient[0]
        scores[Scores.DSCP_COMP] = d_per_patient[1]
        scores[Scores.DSCP_CORE] = d_per_patient[2]
        scores[Scores.DSCP_EN] = d_per_patient[3]

        scores[Scores.PRECISION] = p_per_patient[0]
        scores[Scores.PRECISION_COMP] = p_per_patient[1]
        scores[Scores.PRECISION_CORE] = p_per_patient[2]
        scores[Scores.PRECISION_EN] = p_per_patient[3]
        scores[Scores.SENSITIVITY] = sen_per_patient[0]
        scores[Scores.SENSITIVITY_COMP] = sen_per_patient[1]
        scores[Scores.SENSITIVITY_CORE] = sen_per_patient[2]
        scores[Scores.SENSITIVITY_EN] = sen_per_patient[3]
        scores[Scores.SPECIFICITY] = spec_per_patient[0]
        scores[Scores.SPECIFICITY_COMP] = spec_per_patient[1]
        scores[Scores.SPECIFICITY_CORE] = spec_per_patient[2]
        scores[Scores.SPECIFICITY_EN] = spec_per_patient[3]

        if self._mode == TrainingModes.BRATS_SEGMENTATION or self._mode == TrainingModes.BRATS_SEGMENTATION_TV_PSEUDO_PATIENT:
            scores[Scores.VALSCORE] = scores[Scores.DSCP]
        else:
            scores[Scores.VALSCORE] = - scores[Scores.LOSS]

        return shape, scores

    @staticmethod
    def store_prediction(name, mode, path, batch_x, batch_y, batch_tv, prediction, gt_is_one_hot=1):
        if mode == TrainingModes.TVFLOW_SEGMENTATION:
            img = dutil.combine_img_prediction_tvclustering(data=batch_x, gt=batch_y, tv=batch_tv, pred=prediction)
        else:
            img = dutil.combine_img_prediction(batch_x, batch_y, prediction,
                                               mode=0 if gt_is_one_hot == 1 else 1)
        ioutil.save_image(img, "%s/%s.jpg" % (path, name))