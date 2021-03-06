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

import tensorflow as tf


def weight_variable(shape, stddev=0.1, name="weight", trainable=True):
    """
    Creates a tf weight variable for a convolutional layer

    :param shape: shape of the variable to create
    :param stddev: standard deviation of the newly initialized variable
    :param name: name of the variable
    :param trainable: bool indication whether the variable should be trainable or frozen during training

    :returns a tf variable of shape [shape]
    """
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name, trainable=trainable)


def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc", trainable=True):
    """
    Creates a tf weight variable for a de-convolutional layer

    :param shape: shape of the variable to create
    :param stddev: standard deviation of the newly initialized variable
    :param name: name of the variable
    :param trainable: bool indication whether the variable should be trainable or frozen during training

    :returns a tf variable of shape [shape]
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name, trainable=trainable)


def bias_variable(shape, name="bias", trainable=True):
    """
    Creates a tf bias variable for a convolutional layer

    :param shape: shape of the variable to create
    :param name: name of the variable
    :param trainable: bool indication whether the variable should be trainable or frozen during training

    :returns a tf variable of shape [shape] initialized with 0.1
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name, trainable=trainable)


def conv2d(x, W, b, keep_prob, padding='VALID', bn=False, spatial_dropout=True):
    """
    Defines a convolution operation on tf tensor x

    :param x: input tf tensor
    :param W: weight variable (tf tensor)
    :param b: bias variable (tf tensor)
    :param keep_prob: tf tensor for the dropout. None if not using dropout
    :param padding: mode to pad the convolution. Use "SAME" to use zero padding
    :param bn: bool indicating whether to use batch normalization after the convolution operator
    :param spatial_dropout: bool indicating whether use channel dropout instead of neuron dropout
    :returns a tf tensor holding the output of the convolution
    """
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        cs = tf.shape(conv_2d_b)
        ds = [cs[0], 1, 1, cs[3]]
        if bn:
            conv_2d_b = tf.layers.batch_normalization(conv_2d_b)
        if keep_prob is not None:
            conv_2d_b = tf.nn.dropout(conv_2d_b, keep_prob=keep_prob, noise_shape=ds if spatial_dropout else None)
        return conv_2d_b


def deconv2d(x, W, stride, keep_prob, spatial_dropout=True):
    """
    Defines an inverse convolution operation on tf tensor x

    :param x: input tf tensor
    :param W: weight variable (tf tensor)
    :param keep_prob: tf tensor for the dropout. None if not using dropout
    :param stride: stride of the deconvolution
    :param spatial_dropout: bool indicating whether use channel dropout instead of neuron dropout
    :returns a tf tensor holding the output of the de convolution
    """
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        deconv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID',
                                        name="conv2d_transpose")
        cs = tf.shape(deconv)
        ds = [cs[0], 1, 1, cs[3]]
        if keep_prob is not None:
            deconv = tf.nn.dropout(deconv, keep_prob=keep_prob, noise_shape=ds if spatial_dropout else None)
        return deconv


def max_pool(x, n, keep_prob, spatial_dropout=True):
    """
    Defines an inverse convolution operation on tf tensor x

    :param x: input tf tensor
    :param n: pool size
    :param keep_prob: tf tensor for the dropout. None if not using dropout
    :param spatial_dropout: bool indicating whether use channel dropout instead of neuron dropout
    :returns a tf tensor holding the output of the max pooling
    """
    mp = tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')
    cs = tf.shape(mp)
    ds = [cs[0], 1, 1, cs[3]]
    if keep_prob is not None:
        mp = tf.nn.dropout(mp, keep_prob, noise_shape=ds if spatial_dropout else None)
    return mp


def crop_and_concat(x1, x2, keep_prob=1.0, spatial_dropout=True):
    """
    Concats two  4d tensors on the last axis . crops x1 to the shape of x2 if shape does not match

    :param x1: first input tf tensor
    :param x2: second input tf tensor
    :param keep_prob: tf tensor for the dropout after the operation. None if not using dropout
    :param spatial_dropout: bool indicating whether use channel dropout instead of neuron dropout
    :returns a tf tensor holding the output of the concatination
    """
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        if x1_shape == x1_shape:
            conc = tf.concat([x1, x2], 3)
        else:
            # offsets for the top left corner of the crop
            offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
            size = [-1, x2_shape[1], x2_shape[2], -1]
            x1_crop = tf.slice(x1, offsets, size)
            conc = tf.concat([x1_crop, x2], 3)
        if keep_prob is not None:
            cs = tf.shape(conc)
            ds = [cs[0], 1, 1, cs[3]]
            conc = tf.nn.dropout(conc, keep_prob=keep_prob, noise_shape=ds if spatial_dropout else None)
        return conc
