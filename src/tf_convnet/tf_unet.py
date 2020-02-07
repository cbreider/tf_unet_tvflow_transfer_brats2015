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


def create_2d_unet(x, keep_prob, channels, n_class, n_layers=5, features_root=64, filter_size=3, pool_size=2,
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

    logging.info("Building Unet with, "
                 "Layers {layers}, "
                 "Root feature size {features}, "
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
            size -= 2 * 2 * (filter_size // 2)  # valid conv

    last_feature_map = in_node

    # Output Map
    with tf.name_scope("output_map"):
        weight = weight_variable([1, 1, features_root, n_class], stddev, trainable=True)
        bias = bias_variable([n_class], name="bias", trainable=True)
        conv = conv2d(in_node, weight, bias, tf.constant(1.0))
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
                output_map = x_image - output_map
            else:
                output_map = output_map + x_image
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

    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)

    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)

    return output_map, variables, int(in_size - size), [weight, bias], last_feature_map
