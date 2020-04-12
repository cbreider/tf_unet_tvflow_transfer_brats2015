"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020

adapted from jakeret
source: https://github.com/jakeret/tf_unet.git
"""


from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import logging
import tensorflow as tf
import src.utils.tf_utils as tfu
from src.utils.enum_params import Activation_Func
from collections import OrderedDict
from src.tf_convnet.layers import (weight_variable, weight_variable_devonc, bias_variable,
                                   conv2d, deconv2d, max_pool, crop_and_concat)


def create_2d_unet(x, keep_prob_conv1, keep_prob_conv2, keep_prob_pool, keep_prob_tconv, keep_prob_concat,
                   channels, n_class, n_layers=5,
                   features_root=64, filter_size=3, pool_size=2, summaries=True, use_padding=False, bn=False,
                   trainable_layers=None, layers_to_restore=None, add_residual_layer=False, use_scale_image_as_gt=False,
                   act_func_out=Activation_Func.RELU):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob_conv: dropout probability tensor for convolutions
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param n_layers: number of layers in the net, default 5
    :param features_root: number of features in the first layer, default 64
    :param filter_size: size of the convolution filter, default 3
    :param pool_size: size of the max pooling operation, default 2
    :param summaries: Flag if summaries should be created, default True
    :param trainable_layers: Dictionary of layer to train or not. None to train complete network
    :param use_padding: True to use padding and preserve image sizes. in_size=out_size, default False
    :param bn: True to use batch normalization, default False
    :param add_residual_layer: Add skip layer from input to output new_out = out + in
    :param use_scale_image_as_gt: Use scale layer from tv as gt (only for tv learning) default False
    :param act_func_out: Activation function for out map
    """

    logging.info("Building Unet with: "
                 "Layers= {layers}, "
                 "Root feature size= {features}, "
                 "Filter size= {filter_size}x{filter_size}, "
                 "Pool size= {pool_size}x{pool_size}".format(layers=n_layers,
                                                             features=features_root,
                                                             filter_size=filter_size,
                                                             pool_size=pool_size))

    in_node = x

    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    variables_to_restore = []
    trainable_variables = []
    variables = []

    train_all = True
    if trainable_layers:
        train_all = False

    if use_padding:
        padding = "SAME" # pad convolution outputs to original map size
    else:
        padding = "VALID" # no padding

    in_size = 1000
    size = in_size
    # down layers
    for layer in range(0, n_layers):
        l_name = "down_conv_{}".format(str(layer))
        l_trainable = True if train_all else trainable_layers[l_name]
        if not l_trainable:
            logging.info("Freezing layer {}".format(l_name))

        restore_layer = False if layers_to_restore is None else layers_to_restore[l_name]
        if restore_layer:
            logging.info("Restoring layer {} from checkpoint".format(l_name))

        with tf.name_scope(l_name):
            features = 2 ** layer * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, channels, features], stddev, name="w1",
                                     trainable=l_trainable)
            else:
                w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev, name="w1",
                                     trainable=l_trainable)

            w2 = weight_variable([filter_size, filter_size, features, features], stddev, name="w2",
                                 trainable=l_trainable)
            b1 = bias_variable([features], name="b1", trainable=l_trainable)
            b2 = bias_variable([features], name="b2", trainable=l_trainable)

            conv1 = conv2d(in_node, w1, b1, 1.0 - (keep_prob_conv1 / float(layer+1)), padding=padding, bn=bn)
            tmp_h_conv = tf.nn.relu(conv1)
            conv2 = conv2d(tmp_h_conv, w2, b2, 1.0 - (keep_prob_conv2 / float(layer+1)), padding=padding, bn=bn)
            dw_h_convs[layer] = tf.nn.relu(conv2)

            convs.append((conv1, conv2))

            variables.append(w1)
            variables.append(w2)
            variables.append(b1)
            variables.append(b2)

            if restore_layer:
                variables_to_restore.append(w1)
                variables_to_restore.append(w2)
                variables_to_restore.append(b1)
                variables_to_restore.append(b2)
            if l_trainable:
                trainable_variables.append(w1)
                trainable_variables.append(w2)
                trainable_variables.append(b1)
                trainable_variables.append(b2)

            size -= 2 * 2 * (filter_size // 2) # valid conv
            if layer < n_layers - 1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size, 1.0)
                in_node = pools[layer]
                size /= pool_size

    in_node = dw_h_convs[n_layers - 1]

    # up layers
    for layer in range(n_layers - 2, -1, -1):
        l_name = "up_conv_{}".format(str(layer))
        l_trainable_conv = True if train_all else trainable_layers[l_name][1]
        l_trainable_upconv = True if train_all else trainable_layers[l_name][0]
        if not l_trainable_conv:
            logging.info("Freezing layer {} convolution block".format(l_name))
        if not l_trainable_upconv:
            logging.info("Freezing layer {} up convolution".format(l_name))

        restore_layer_conv = False if layers_to_restore is None else layers_to_restore[l_name][1]
        if restore_layer_conv:
            logging.info("Restoring conv layer {} from checkpoint".format(l_name))
        restore_layer_up = False if layers_to_restore is None else layers_to_restore[l_name][0]
        if restore_layer_up:
            logging.info("Restoring up conv layer {} from checkpoint".format(l_name))

        with tf.name_scope(l_name):
            features = 2 ** (layer + 1) * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))

            wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd",
                                        trainable=l_trainable_upconv)
            bd = bias_variable([features // 2], name="bd", trainable=l_trainable_upconv)
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size, 1.0) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv, keep_prob=keep_prob_concat)
            deconv[layer] = h_deconv_concat

            w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1",
                                 trainable=l_trainable_conv)
            w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2",
                                 trainable=l_trainable_conv)
            b1 = bias_variable([features // 2], name="b1", trainable=l_trainable_conv)
            b2 = bias_variable([features // 2], name="b2", trainable=l_trainable_conv)

            conv1 = conv2d(h_deconv_concat, w1, b1, 1.0 - (keep_prob_conv1 / float(layer+1)), padding=padding, bn=bn)
            h_conv = tf.nn.relu(conv1)

            conv2 = conv2d(h_conv, w2, b2, 1.0 - (keep_prob_conv2 / float(layer+1)), padding=padding, bn=bn)
            in_node = tf.nn.relu(conv2)
            up_h_convs[layer] = in_node

            variables.append(w1)
            variables.append(w2)
            variables.append(b1)
            variables.append(b2)
            variables.append(wd)
            variables.append(bd)

            if restore_layer_up:
                variables_to_restore.append(wd)
                variables_to_restore.append(bd)
            if restore_layer_conv:
                variables_to_restore.append(w1)
                variables_to_restore.append(w2)
                variables_to_restore.append(b1)
                variables_to_restore.append(b2)
            if l_trainable_upconv:
                trainable_variables.append(wd)
                trainable_variables.append(bd)
            if l_trainable_conv:
                trainable_variables.append(w1)
                trainable_variables.append(w2)
                trainable_variables.append(b1)
                trainable_variables.append(b2)

            convs.append((conv1, conv2))

            size *= pool_size
            size -= 2 * 2 * (filter_size // 2)  # valid conv

    last_feature_map = in_node

    # Output Map
    with tf.name_scope("output_map"):
        l_trainable = True if train_all else trainable_layers["classifier"]
        if not l_trainable:
            logging.info("Freezing layer {}".format("classifier"))
        restore_layer = False if layers_to_restore is None else layers_to_restore["classifier"]
        if restore_layer:
            logging.info("Restoring layer {} from checkpoint".format("classifier"))
        weight = weight_variable([1, 1, features_root, n_class], stddev, trainable=l_trainable)
        bias = bias_variable([n_class], name="bias", trainable=l_trainable)
        conv = conv2d(in_node, weight, bias, tf.constant(1.0))

        variables.append(weight)
        variables.append(bias)

        if restore_layer:
            variables_to_restore.append(weight)
            variables_to_restore.append(bias)
        if l_trainable:
            trainable_variables.append(weight)
            trainable_variables.append(bias)

        if act_func_out == Activation_Func.NONE or n_class == 1:  # in binary sigmoid is added within cs loss
            output_map = conv
        elif act_func_out == Activation_Func.RELU:
            output_map = tf.nn.relu(conv)
        elif act_func_out == Activation_Func.SIGMOID:
            output_map = tf.nn.sigmoid(conv)
        else:
            raise ValueError()
        if add_residual_layer:
            if not padding == 'SAME':
                raise ValueError("Residual Layer only possible with padding to preserve same size feature maps")
            if use_scale_image_as_gt:
                output_map = x - output_map
            else:
                output_map = output_map + x
        up_h_convs["out"] = output_map

    if summaries:
        with tf.name_scope("summaries"):
            for i, (c1, c2) in enumerate(convs):
                tf.summary.image('summary_conv_%02d_01' % i, tfu.get_image_summary(c1))
                tf.summary.image('summary_conv_%02d_02' % i, tfu.get_image_summary(c2))

            for k in pools.keys():
                tf.summary.image('summary_pool_%02d' % k, tfu.get_image_summary(pools[k]))

            for k in deconv.keys():
                tf.summary.image('summary_deconv_concat_%02d' % k, tfu.get_image_summary(deconv[k]))

            for k in dw_h_convs.keys():
                tf.summary.histogram("dw_convolution_%02d" % k + '/activations', dw_h_convs[k])

            for k in up_h_convs.keys():
                tf.summary.histogram("up_convolution_%s" % k + '/activations', up_h_convs[k])

    return [output_map, last_feature_map, variables_to_restore, trainable_variables, variables, int(in_size - size)]
