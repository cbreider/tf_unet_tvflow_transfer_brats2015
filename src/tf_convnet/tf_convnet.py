"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import logging
import src.tf_convnet.tf_unet as tf_unet
import tensorflow as tf
from datetime import datetime
from src.utils.enum_params import Cost, RestoreMode
import src.utils.tf_utils as tfu
from configuration import ConvNetParams
from src.tf_convnet.layers import (weight_variable, weight_variable_devonc, bias_variable,
                                   conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax,
                                   cross_entropy)


class ConvNetModel(object):

    def __init__(self, convnet_config, create_summaries=True):
        """
        A unet implementation

        :param convnet_config: config params for convnet
        :param create_summaries: Flag if summaries should be created
            """
        self._convnet_config = convnet_config  # type: ConvNetParams
        self._n_class = self._convnet_config.nr_of_classes
        self._n_channels = self._convnet_config.nr_input_channels
        self.summaries = create_summaries
        self.cost_function = self._convnet_config.cost_function
        self._n_layers = self._convnet_config.num_layers
        self._features_root = self._convnet_config.feat_root
        self._filter_size = self._convnet_config.filter_size
        self._pool_size = self._convnet_config.pool_size
        self._freeze_down_layers = self._convnet_config.freeze_down_layers
        self._freeze_up_layers = self._convnet_config.freeze_up_layers
        self._use_padding = self._convnet_config.padding
        self._add_residual_layer = self._convnet_config.add_residual_layer
        self._batch_norm = self._convnet_config.batch_normalization
        self._use_scale_image_as_gt = self._convnet_config.use_scale_as_gt
        self._activation_func_out = self._convnet_config.activation_func_out
        self._tv_regularizer = self._convnet_config.tv_regularizer
        self._class_weights = self._convnet_config.class_weights
        self._regularizer = self._convnet_config.regularizer
        self._max_tv_value = self._convnet_config.max_tv_value
        self._two_classes_are_binary = self._convnet_config.two_classe_as_binary

        self.x = tf.placeholder("float", shape=[None, None, None, self._n_channels], name="x")
        self.y = tf.placeholder("float", shape=[None, None, None, self._n_class], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_probability")  # dropout (keep probability)

        self.logits, self.variables, self.offset, self.out_vars, lfs = \
            tf_unet.create_2d_unet(x=self.x,
                                   keep_prob=self.keep_prob,
                                   channels=self._n_channels,
                                   n_class=self._n_class,
                                   n_layers=self._n_layers,
                                   filter_size=self._filter_size,
                                   pool_size=self._pool_size,
                                   summaries=self.summaries,
                                   freeze_down_layers=self._freeze_down_layers,
                                   freeze_up_layers=self._freeze_up_layers,
                                   use_padding=self._use_padding,
                                   bn=self._batch_norm,
                                   add_residual_layer=self._add_residual_layer,
                                   use_scale_image_as_gt=self._use_scale_image_as_gt,
                                   act_func_out=self._activation_func_out)

        self.cost = self._get_cost()

        self.gradients_node = tf.gradients(self.cost, self.variables)
        self.last_feature_map = lfs

        with tf.name_scope("cross_entropy"):
            if self.cost_function == Cost.MSE:
                self.cross_entropy = tf.constant(0)  # cross entropy for regression useless
            else:
                self.cross_entropy = tfu.get_cross_entropy(logits=tf.reshape(self.logits, [-1, self._n_class]),
                                                           y=tf.reshape(self.y, [-1, self._n_class]),
                                                           n_class=self._n_class,
                                                           weights=None)
        with tf.name_scope("results"):
            if self.cost_function == Cost.MSE:
                self.correct_pred = tf.constant(0)  # makes no sense for regression
                self.predicter = self.logits
                self.error = tf.math.divide(tf.math.reduce_sum(tf.math.squared_difference(self.predicter, self.y)),
                                            tf.cast(tf.size(self.y), tf.float32))
                self.error = tf.math.divide(self.error,
                                            tf.math.square(tf.constant(self._max_tv_value)))
                self.error_rate = tf.math.multiply(tf.constant(100.0), self.error)
                self.accuracy = tf.constant(1.0) - self.error
                self.dice = tf.constant(0)
            else:
                if self._n_class > 1:
                    self.predicter = tf.nn.softmax(self.logits, axis=3)
                    self.pred_slice = tf.cast(tf.argmax(self.predicter, axis=3), tf.float32)
                    self.y_slice = tf.cast(tf.argmax(self.y, axis=3), tf.float32)
                    self.correct_pred = tf.equal(self.pred_slice, self.y_slice)
                else:
                    self.predicter = tf.cast(tf.nn.sigmoid(self.logits) > 0.5, tf.float32)
                    self.correct_pred = tf.equal(self.predicter, self.y)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
                self.error = tf.constant(1.0) - self.accuracy
                self.error_rate = tf.math.multiply(self.error, tf.constant(100.0))
                self.dice = tfu.get_dice_score(pred=self.predicter, y=self.y, eps=1e-5,
                                               binary=self._two_classes_are_binary)

    def _get_cost(self):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient
        """
        with tf.name_scope("cost"):
            flat_logits = tf.reshape(self.logits, [-1, self._n_class])
            flat_labels = tf.reshape(self.y, [-1, self._n_class])
            if self.cost_function == Cost.CROSS_ENTROPY:
                loss = tfu.get_cross_entropy(logits=flat_logits, y=flat_labels, n_class=self._n_class,
                                             weights=self._class_weights)

            elif self.cost_function == Cost.DICE_COEFFICIENT:
                loss = 1.0 - tfu.get_dice_score(pred=self.logits, y=self.y, eps=1e-5, binary=self._two_classes_are_binary)

            elif self.cost_function == Cost.MSE:
                loss = tf.losses.mean_squared_error(pixel_wise_softmax(self.logits), flat_labels)

            elif self.cost_function == Cost.TV:
                loss = tf.losses.mean_squared_error(flat_logits, flat_labels)
                tv = tf.reduce_sum(tf.image.total_variation(self.logits))
                loss += self._tv_regularizer * tv
            else:
                raise ValueError("Unknown cost function: " % self.cost_function.name)

            if self._regularizer is not None:
                regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
                loss += (self._regularizer * regularizers)

            return loss

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})

        return prediction

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
        """
        logging.info('{} Restoring from tf checkpoint: {}'.format(datetime.now(), model_path))
        if restore_mode == RestoreMode.COMPLETE_SESSION:
            logging.info('{} Resuming complete session: {}'.format(datetime.now(), model_path))
            saver = tf.train.Saver()
        elif restore_mode == RestoreMode.COMPLETE_NET:
            logging.info('{} Restoring Complete Net: {}'.format(datetime.now(), model_path))
            saver = tf.train.Saver(self.variables + self.out_vars)
        elif restore_mode == RestoreMode.ONLY_BASE_NET:
            logging.info('{} Restoring only Bases Net: {}'.format(datetime.now(), model_path))
            saver = tf.train.Saver(self.variables)
        else:
            raise ValueError()

        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)


