"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019

adapted from jakeret
source: https://github.com/jakeret/tf_unet.git
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import logging
import os
import tensorflow as tf
from datetime import datetime
from src.utils.enum_params import Cost, Activation_Func, RestoreMode
from collections import OrderedDict
from src.tf_unet.layers import (weight_variable, weight_variable_devonc, bias_variable,
                                conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax,
                                cross_entropy)


def create_conv_net(x, keep_prob, channels, n_class, n_layers=5, features_root=64, filter_size=3, pool_size=2,
                    summaries=True, freeze_down_layers=False, freeze_up_layers=False, use_padding=False, bn=False,
                    add_residual_layer=False, use_scale_image_as_gt=False, act_func_out=Activation_Func.RELU):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param n_layers: number of layers in the net, default 5
    :param features_root: number of features in the first layer, default 64
    :param filter_size: size of the convolution filter, default 3
    :param pool_size: size of the max pooling operation, default 2
    :param summaries: Flag if summaries should be created, default True
    :param freeze_down_layers: True to freeze layers in decoder, default False
    :param freeze_up_layers: True to freeze layers in decoder, default False
    :param use_padding: True touse padding and preserve image sizes. in_size=out_size, default False
    :param bn: True to use batch normalization, default False
    :param add_residual_layer: Add skip layer from input to output new_out = out + in
    :param use_scale_image_as_gt: Use scale layer from tv as gt (only for tv learning) default False
    :param act_func_out: Activation function for out map
    """

    logging.info("Building Unet with,"
                 "Layers {layers}, "
                 "features {features}, "
                 "filter size {filter_size}x{filter_size}, "
                 "pool size: {pool_size}x{pool_size}".format(layers=n_layers,
                                                             features=features_root,
                                                             filter_size=filter_size,
                                                             pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image

    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    variables = []

    if freeze_down_layers:
        logging.info("Freezing down Layers!")
        trainable_down = False
    else:
        trainable_down = True

    if freeze_up_layers:
        logging.info("Freezing up Layers!")
        trainable_up = False
    else:
        trainable_up = True

    if use_padding:
        padding = "SAME" # pad convolution outputs to original map size
    else:
        padding = "VALID" # no padding

    in_size = 1000
    size = in_size
    # down layers
    for layer in range(0, n_layers):
        with tf.name_scope("down_conv_{}".format(str(layer))):
            features = 2 ** layer * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, channels, features], stddev, name="w1",
                                     trainable=trainable_down)
            else:
                w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev, name="w1",
                                     trainable=trainable_down)

            w2 = weight_variable([filter_size, filter_size, features, features], stddev, name="w2",
                                 trainable=trainable_down)
            b1 = bias_variable([features], name="b1", trainable=trainable_down)
            b2 = bias_variable([features], name="b2", trainable=trainable_down)

            conv1 = conv2d(in_node, w1, b1, keep_prob, padding=padding, bn=bn)
            tmp_h_conv = tf.nn.relu(conv1)
            conv2 = conv2d(tmp_h_conv, w2, b2, keep_prob, padding=padding, bn=bn)
            dw_h_convs[layer] = tf.nn.relu(conv2)

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size -= 2 * 2 * (filter_size // 2) # valid conv
            if layer < n_layers - 1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= pool_size

    in_node = dw_h_convs[n_layers - 1]

    # up layers
    for layer in range(n_layers - 2, -1, -1):
        with tf.name_scope("up_conv_{}".format(str(layer))):
            features = 2 ** (layer + 1) * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))

            wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd",
                                        trainable=trainable_up)
            bd = bias_variable([features // 2], name="bd", trainable=trainable_up)
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat

            w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1",
                                 trainable=trainable_up)
            w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2",
                                 trainable=trainable_up)
            b1 = bias_variable([features // 2], name="b1", trainable=trainable_up)
            b2 = bias_variable([features // 2], name="b2", trainable=trainable_up)

            conv1 = conv2d(h_deconv_concat, w1, b1, keep_prob, padding=padding, bn=bn)
            h_conv = tf.nn.relu(conv1)
            conv2 = conv2d(h_conv, w2, b2, keep_prob, padding=padding, bn=bn)
            in_node = tf.nn.relu(conv2)
            up_h_convs[layer] = in_node

            variables.append(wd)
            variables.append(bd)
            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size *= pool_size
            size -= 2 * 2 * (filter_size // 2) # valid conv

    # Output Map
    with tf.name_scope("output_map"):
        weight = weight_variable([1, 1, features_root, n_class], stddev, trainable=True)
        bias = bias_variable([n_class], name="bias", trainable=True)
        conv = conv2d(in_node, weight, bias, tf.constant(1.0))
        if act_func_out == Activation_Func.RELU:
            output_map = tf.nn.relu(conv)
        elif act_func_out == Activation_Func.SIGMOID:
            output_map = tf.nn.sigmoid(conv)
        elif act_func_out == Activation_Func.NONE:
            output_map = conv
        else:
            raise ValueError()
        if add_residual_layer:
            if not padding == 'SAME':
                raise ValueError("Residual Layer only possible with padding to preserve same size feature maps")
            if use_scale_image_as_gt:
                output_map = x_image - output_map
            else:
                output_map = output_map + x_image
        up_h_convs["out"] = output_map

    if summaries:
        with tf.name_scope("summaries"):
            for i, (c1, c2) in enumerate(convs):
                tf.summary.image('summary_conv_%02d_01' % i, get_image_summary(c1))
                tf.summary.image('summary_conv_%02d_02' % i, get_image_summary(c2))

            for k in pools.keys():
                tf.summary.image('summary_pool_%02d' % k, get_image_summary(pools[k]))

            for k in deconv.keys():
                tf.summary.image('summary_deconv_concat_%02d' % k, get_image_summary(deconv[k]))

            for k in dw_h_convs.keys():
                tf.summary.histogram("dw_convolution_%02d" % k + '/activations', dw_h_convs[k])

            for k in up_h_convs.keys():
                tf.summary.histogram("up_convolution_%s" % k + '/activations', up_h_convs[k])

    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)

    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)

    return output_map, variables, int(in_size - size), [weight, bias]


class Unet(object):
    """
    A unet implementation

    :param n_channels: number of channels in the input image
    :param n_class: number of output labels
    :param cost_function: (optional) name of the cost function. Default is 'cross_entropy'
    :param n_layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
        """

    def __init__(self, n_channels, n_class, cost_function=Cost.CROSS_ENTROPY, summaries=True, class_weights=None,
                 regularizer=None, n_layers=5, keep_prob=0.5, features_root=64, filter_size=3, pool_size=2,
                 freeze_down_layers=False, freeze_up_layers=False, use_padding=False, batch_norm=False,
                 add_residual_layer=False, use_scale_image_as_gt=False, max_gt_value=1.0,
                 act_func_out=Activation_Func.RELU):

        self.n_class = n_class
        self.n_channels = n_channels
        self.summaries = summaries
        self.cost_function = cost_function

        self.x = tf.placeholder("float", shape=[None, None, None, n_channels], name="x")
        self.y = tf.placeholder("float", shape=[None, None, None, n_class], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_probability")  # dropout (keep probability)

        logits, self.variables, self.offset, self.out_vars = create_conv_net(x=self.x,
                                                              keep_prob=keep_prob,
                                                              channels=n_channels,
                                                              n_class=n_class,
                                                              n_layers=n_layers,
                                                              features_root=features_root,
                                                              filter_size=filter_size,
                                                              pool_size=pool_size,
                                                              summaries=summaries,
                                                              freeze_down_layers=freeze_down_layers,
                                                              freeze_up_layers=freeze_up_layers,
                                                              use_padding=use_padding,
                                                              bn=batch_norm,
                                                              add_residual_layer=add_residual_layer,
                                                              use_scale_image_as_gt=use_scale_image_as_gt,
                                                              act_func_out=act_func_out)

        self.cost = self._get_cost(logits=logits,
                                   cost_function=cost_function,
                                   class_weights=class_weights,
                                   regularizer=regularizer)

        self.gradients_node = tf.gradients(self.cost, self.variables)

        with tf.name_scope("cross_entropy"):
            if cost_function == Cost.MSE:
                self.cross_entropy = tf.constant(0) # cross entropy for regression useless
            else:
                self.cross_entropy = cross_entropy(tf.reshape(self.y, [-1, n_class]),
                                                   tf.reshape(pixel_wise_softmax(logits), [-1, n_class]))

        with tf.name_scope("results"):
            if cost_function == Cost.MSE:
                self.correct_pred = tf.constant(0)  # makes no sense for regression
                self.predicter = logits
                self.error = tf.math.divide(tf.math.reduce_sum(tf.math.squared_difference(self.predicter, self.y)),
                                            tf.cast(tf.size(self.y), tf.float32))
                self.error = tf.math.divide(self.error,
                                            tf.math.square(tf.constant(max_gt_value)))
                self.error_rate = tf.math.multiply(tf.constant(100.0), self.error)
                self.accuracy = tf.constant(1.0) - self.error
                self.dice = tf.constant(0)
            else:
                self.predicter = pixel_wise_softmax(logits)
                if self.n_class == 2:
                    self.pred_slice = tf.cast(tf.argmax(self.predicter, axis=3), tf.float32)
                    self.y_slice = tf.cast(tf.argmax(self.y, axis=3), tf.float32)
                    self.correct_pred = tf.equal(self.pred_slice, self.y_slice)
                else:
                    self.pred_slice = tf.argmax(self.predicter, axis=3)
                    self.pred_slice = tf.one_hot(self.pred_slice, depth=self.n_class)
                    self.y_slice = self.y
                    self.correct_pred = tf.equal(tf.argmax(self.pred_slice, axis=3), tf.argmax(self.y_slice, axis=3))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
                self.error = tf.constant(1.0) - self.accuracy
                self.error_rate = tf.math.multiply(self.error, tf.constant(100.0))
                eps = 1e-5
                intersection = tf.reduce_sum(self.pred_slice * self.y_slice)
                union = eps + tf.reduce_sum(self.pred_slice) + tf.reduce_sum(self.y_slice)
                self.dice = (2 * intersection / union)

    def _get_cost(self, logits, cost_function, class_weights=None, regularizer=None):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.

        :param logits: out put of net
        :param cost_function: name of the cost function. Default is 'cross_entropy'
        :param class_weights: weights for the different classes in case of multi-class imbalance
        :param regularizer: power of the L2 regularizers added to the loss function
        """
        tf_reg = True
        with tf.name_scope("cost"):
            flat_logits = tf.reshape(logits, [-1, self.n_class])
            flat_labels = tf.reshape(self.y, [-1, self.n_class])
            if cost_function == Cost.CROSS_ENTROPY:

                if class_weights is not None:
                    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

                    weight_map = tf.multiply(flat_labels, class_weights)
                    weight_map = tf.reduce_sum(weight_map, axis=1)

                    loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                          labels=flat_labels)
                    weighted_loss = tf.multiply(loss_map, weight_map)

                    loss = tf.reduce_mean(weighted_loss)

                else:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                     labels=flat_labels))
            elif cost_function == Cost.DICE_COEFFICIENT:
                #eps = 1e-5
                #prediction = pixel_wise_softmax(logits)
                #intersection = tf.reduce_sum(prediction * self.y)
                #union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
                #loss = -(2 * intersection / (union))
                smooth = 1e-17
                logits = tf.nn.softmax(logits)
                weights = 1.0 / (tf.reduce_sum(self.y))

                numerator = tf.reduce_sum(self.y * logits)
                numerator = tf.reduce_sum(weights * numerator)

                denominator = tf.reduce_sum(self.y + logits, axis=[0, 1, 2])
                denominator = tf.reduce_sum(weights * denominator)

                loss = 1.0 - 2.0 * (numerator + smooth) / (denominator + smooth)

            elif cost_function == Cost.MSE:
                loss = tf.losses.mean_squared_error(flat_logits, flat_labels)

            else:
                raise ValueError("Unknown cost function: " % cost_function.name)

            if regularizer is not None:
                regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
                loss += (regularizer * regularizers)

            tv = tf.reduce_sum(tf.image.total_variation(logits))
            loss += 0.1 * tv

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
        print('{} Restoring from tf checkpoint: {}'.format(datetime.now(), model_path))
        if restore_mode == RestoreMode.COMPLETE_SESSION:
            print('{} Resuming complete session: {}'.format(datetime.now(), model_path))
            saver = tf.train.Saver()
        elif restore_mode == RestoreMode.COMPLETE_NET:
            print('{} Restoring Complete Net: {}'.format(datetime.now(), model_path))
            saver = tf.train.Saver(self.variables + self.out_vars)
        elif restore_mode == RestoreMode.ONLY_BASE_NET:
            print('{} Restoring only Bases Net: {}'.format(datetime.now(), model_path))
            saver = tf.train.Saver(self.variables)
        else:
            raise ValueError()

        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
