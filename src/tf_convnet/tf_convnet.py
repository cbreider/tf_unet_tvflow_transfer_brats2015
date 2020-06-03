"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020

"""
from __future__ import print_function, division, absolute_import, unicode_literals

import logging
import src.tf_convnet.tf_unet as tf_unet
import tensorflow as tf
from datetime import datetime
from src.utilities.enum_params import Cost, RestoreMode, TrainingModes, ConvNetType
import src.utilities.tf_utils as tfu
from configuration import Configuration


class ConvNetModel(object):
    """
    :
    """
    def __init__(self, convnet_config, mode, create_summaries=True):
        """
        A unet implementation

        :param convnet_config: config params for convnet
        :param mode: Training mode of the convnet
        :param create_summaries: Flag if summaries should be created
            """
        self._convnet_config = convnet_config  # type: Configuration
        self._n_class = self._convnet_config.nr_classes
        self._n_channels = self._convnet_config.nr_input_channels
        self.summaries = create_summaries
        self.cost_function = self._convnet_config.cost_function
        self._n_layers = self._convnet_config.num_layers
        self._features_root = self._convnet_config.feat_root
        self._filter_size = self._convnet_config.filter_size
        self._pool_size = self._convnet_config.pool_size
        self._trainable_layers = self._convnet_config.trainable_layers
        self._use_padding = self._convnet_config.padding
        self._add_residual_layer = self._convnet_config.add_residual_layer
        self._batch_norm = self._convnet_config.batch_normalization
        self._tv_regularizer = self._convnet_config.tv_regularizer
        self._class_weights_ce = self._convnet_config.class_weights_ce
        self._class_weights_dice = self._convnet_config.class_weights_dice
        self._l2_regularizer = self._convnet_config.lambda_l2_regularizer
        self._l1_regularizer = self._convnet_config.lambda_l1_regularizer
        self._loss_weight = self._convnet_config.cost_weight
        self._restore_layers = self._convnet_config.restore_layers
        self._remove_skip_layers = self._convnet_config.remove_skip_layers
        self._spatial_dropout = self._convnet_config.spatial_dropuout
        self._mode = mode
        self._model_type = self._convnet_config.conv_net_type # Type ConvNetType
        self.l1regularizers = tf.constant(-1.0)
        self.l2regularizers = tf.constant(-1.0)

        # declare placeholders for input output and dropout probabilities. The actual values will be passes during training
        self.x = tf.placeholder("float", shape=[None, None, None, self._n_channels], name="x")
        self.y = tf.placeholder("float", shape=[None, None, None, self._n_class], name="y")
        # dropout (keep probability) after fist convolution of each layer
        self.keep_prob_conv1 = tf.placeholder(tf.float32, name="dropout_probability_conv1")
        # dropout (keep probability) after second convolution of each layer
        self.keep_prob_conv2 = tf.placeholder(tf.float32, name="dropout_probability_conv2")  # dropout (keep probability)
        # dropout (keep probability) after pooling operation of each layer (encoder)
        self.keep_prob_pool = tf.placeholder(tf.float32, name="dropout_probability_pool")  # dropout (keep probability)
        # dropout (keep probability) after transpose convolution of each layer (decoder)
        self.keep_prob_tconv = tf.placeholder(tf.float32, name="dropout_probability_tconv")  # dropout (keep probability)
        # dropout (keep probability) after skip and concating each layer (decoder)
        self.keep_prob_concat = tf.placeholder(tf.float32, name="dropout_probability_concat")  # dropout (keep probability)

        # create model
        if self._model_type == ConvNetType.U_NET_2D:
            [self.logits, self.last_feature_map, self.variables_to_restore, self.trainable_variables, self.variables,
            self.offset] = tf_unet.create_2d_unet(x=self.x, keep_prob_conv1=self.keep_prob_conv1,
                                                  keep_prob_conv2=self.keep_prob_conv2,
                                                  keep_prob_pool=self.keep_prob_pool,
                                                  keep_prob_tconv=self.keep_prob_tconv,
                                                  keep_prob_concat=self.keep_prob_concat,
                                                  spatial_dropout=self._spatial_dropout, nr_channels=self._n_channels,
                                                  n_class=self._n_class, n_layers=self._n_layers,
                                                  filter_size=self._filter_size,  pool_size=self._pool_size,
                                                  summaries=self.summaries, trainable_layers=self._trainable_layers,
                                                  use_padding=self._use_padding, bn=self._batch_norm,
                                                  add_residual_layer=self._add_residual_layer,
                                                  features_root=self._features_root,
                                                  layers_to_restore=self._restore_layers,
                                                  remove_skip_layers=self._remove_skip_layers)
        # get network loss
        self.cost = self._get_cost()

        # get gradients for tf summary
        self.gradients_node = tf.gradients(self.cost, self.variables)

        # get results scores
        with tf.name_scope("results"):
            # initialze default as -1. Will not be consired for the summary
            self.dice_core = tf.constant(-1.)
            self.dice_complete = tf.constant(-1.)
            self.dice_enhancing = tf.constant(-1.)
            self.iou_coe = tf.constant(-1.)
            self.accuracy = tf.constant(-1.)
            self.dice = tf.constant(-1.)
            self.cross_entropy = tf.constant(-1.)
            if self.cost_function == Cost.MSE:
                # if MSE (regression model) the only score is the mse
                self.predicter = self.logits
                self.error = tf.math.divide(tf.math.reduce_sum(tf.math.squared_difference(self.predicter, self.y)),
                                            tf.cast(tf.size(self.y), tf.float32))
            else:
                # segmentation model
                if self._n_class > 1:
                    if self._mode == TrainingModes.TVFLOW_SEGMENTATION:
                        tf.cast(tf.nn.sigmoid(self.logits) > 0.5, tf.float32) # multi-label
                    # if number of classes is greate 1. softmax instead of sigmoid is used
                    self.predicter = tf.nn.softmax(self.logits, axis=3)
                    self.pred_slice = tf.cast(tf.argmax(self.predicter, axis=3), tf.float32)
                    # get index of predicted class
                    self.y_slice = tf.cast(tf.argmax(self.y, axis=3), tf.float32)
                    # get number of correct predicted values
                    self.correct_pred = tf.equal(self.pred_slice, self.y_slice)
                    if (self._mode == TrainingModes.BRATS_SEGMENTATION or self._mode == TrainingModes.TVFLOW_SEGMENTATION_TV_PSEUDO_PATIENT) and self._n_class == 5:
                        # only if we segment all 5 classes of brats calc three different scores
                        # (complete, core and enhancing)
                        # complete all classes greater 0 (whole tumor)
                        pred_complete = tf.cast(tf.greater(self.pred_slice, 0.), tf.float32)
                        # core class 1 or 3 or 4
                        pred_core = tf.cast(tf.logical_or(tf.logical_or(tf.equal(self.pred_slice, 1.),
                                                                        tf.equal(self.pred_slice, 3.)),
                                                          tf.equal(self.pred_slice, 4.)), tf.float32)
                        # enhancing only class 4
                        pred_enhancing = tf.cast(tf.equal(self.pred_slice, 4.), tf.float32)
                        y_complete = tf.cast(tf.greater(self.y_slice, 0.), tf.float32)
                        y_core = tf.cast(tf.logical_or(tf.logical_or(tf.equal(self.y_slice, 1.),
                                                                     tf.equal(self.y_slice, 3.)),
                                                       tf.equal(self.y_slice, 4.)), tf.float32)
                        y_enhancing = tf.cast(tf.equal(self.y_slice, 4.), tf.float32)
                        # calc dice score for the different masks
                        self.dice_complete = tfu.get_dice_score(pred=pred_complete, y=y_complete, eps=1e-5)
                        self.dice_core = tfu.get_dice_score(pred=pred_core, y=y_core, eps=1e-5)
                        self.dice_enhancing = tfu.get_dice_score(pred=pred_enhancing, y=y_enhancing, eps=1e-5)
                else:
                    # binary segmentation (0 or 1): use pixel wise sigmoid
                    self.predicter = tf.cast(tf.nn.sigmoid(self.logits) > 0.5, tf.float32)
                    self.correct_pred = tf.equal(self.predicter, self.y)
                # calc cross entropy
                self.cross_entropy = tfu.get_cross_entropy(logits=tf.reshape(self.logits, [-1, self._n_class]),
                                                           y=tf.reshape(self.y, [-1, self._n_class]),
                                                           n_class=self._n_class,
                                                           weights=None)
                # intersetcion over union
                self.iou_coe = tfu.get_iou_coe(pre=self.predicter, gt=self.y)
                # Dice score (average over all classes)
                self.dice = tfu.get_dice_score(pred=self.predicter, y=self.y, weights=None)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
                self.error = tf.constant(1.0) - self.accuracy

    def _get_cost(self):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient
        :returns Tf tensor of calculated loss
        """
        logging.info("Cost: {}".format(self.cost_function.name))

        with tf.name_scope("cost"):

            if self.cost_function == Cost.BATCH_DICE_LOG or self.cost_function == Cost.BATCH_DICE_SOFT or \
                    self.cost_function == Cost.BATCH_DICE_SOFT_CE:
                # calculate Dice loss over the complete batch (take batch as pseudo 3d Tensor)
                if self._n_class == 1:
                    # if nr classes is 1 axis 3 has only one component
                    axis = (0, 1, 2, 3)
                else:
                    axis = (0, 1, 2)
            else:
                # compute dice for each slice and take average (normally not used but considered as option)
                if self._n_class == 1:
                    axis = (1, 2, 3)
                else:
                    axis = (1, 2)
            # flatten input and outpout
            flat_logits = tf.reshape(self.logits, [-1, self._n_class])
            flat_labels = tf.reshape(self.y, [-1, self._n_class])

            # cross entropy loss
            if self.cost_function == Cost.CROSS_ENTROPY:
                # if class weights are None cross entropy will not be weighted
                loss = tfu.get_cross_entropy(logits=flat_logits, y=flat_labels, n_class=self._n_class,
                                             weights=self._class_weights_ce)
            # Dice loss
            elif self.cost_function == Cost.DICE_SOFT or self.cost_function == Cost.BATCH_DICE_SOFT:
                loss = 1.0 - tfu.get_dice_loss(logits=self.logits, y=self.y, axis=axis,
                                               weights=self._class_weights_dice, exclude_zero_label=False)
            # Weighted combination of dice and cross entropy
            elif self.cost_function == Cost.DICE_SOFT_CE or self.cost_function == Cost.BATCH_DICE_SOFT_CE:
                loss = self._loss_weight * (1.0 - tfu.get_dice_loss(logits=self.logits, y=self.y, axis=axis,
                                                                    weights=self._class_weights_dice,
                                                                    exclude_zero_label=False))
                loss += (1.0 - self._loss_weight) * tfu.get_cross_entropy(logits=flat_logits, y=flat_labels,
                                                                          n_class=self._n_class,
                                                                          weights=self._class_weights_ce)
            # Dice log loss (-log(dice_score)). Considered to have nicer gradient.
            # But seems to be not realy more valuable in real life
            elif self.cost_function == Cost.DICE_LOG or self.cost_function == Cost.BATCH_DICE_LOG:
                loss = tfu.get_dice_log_loss(self.logits, self.y, axis=axis, exclude_zero_label=False)

            # MSE loss used for regression tasks
            elif self.cost_function == Cost.MSE:
                loss = tf.losses.mean_squared_error(flat_logits, flat_labels)

            # TV loss (MSE + total variation of output as regularizer). Seems to not work very
            elif self.cost_function == Cost.TV:
                loss = tf.losses.mean_squared_error(flat_logits, flat_labels)
                tv = tf.reduce_sum(tf.image.total_variation(self.logits))
                loss += self._tv_regularizer * tv
            else:
                raise ValueError("Unknown cost function: " % self.cost_function.name)

            # if value for l1 or l2 regularizer is given add them to the loss
            if self._l2_regularizer is not None:
                self.l2regularizers = self._l2_regularizer * sum(
                    [tf.nn.l2_loss(variable) for variable in self.variables])
                loss += self.l2regularizers
            if self._l1_regularizer is not None:
                self.l1regularizers = self._l1_regularizer * sum([
                    tf.reduce_sum(tf.abs(variable)) for variable in self.variables])
                loss += self.l1regularizers

            return loss

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path, restore_mode=RestoreMode.COMPLETE_SESSION):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        :param restore_mode: mode of restoring. 1 for complete tf session, 2 or 3 to restore only certain variables
        declared to restore during model creation
        """
        logging.info('{} Restoring from tf checkpoint: {}'.format(datetime.now(), model_path))
        if restore_mode == RestoreMode.COMPLETE_SESSION:
            logging.info('{} Resuming complete session: {}'.format(datetime.now(), model_path))
            saver = tf.train.Saver()
        elif restore_mode == RestoreMode.COMPLETE_NET:
            logging.info('{} Restoring only Network: {}'.format(datetime.now(), model_path))
            saver = tf.train.Saver(self.variables)
        elif restore_mode == RestoreMode.ONLY_GIVEN_VARS:
            logging.info('{} Restoring only Network: {}'.format(datetime.now(), model_path))
            saver = tf.train.Saver(self.variables_to_restore)
        else:
            raise ValueError()

        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)


