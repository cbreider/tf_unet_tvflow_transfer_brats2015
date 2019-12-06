"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019
"""

import h5py
import numpy as np
import tensorflow as tf
from datetime import datetime

"""
This dictionary provides a mapping from the original caffe model to the tf mpdel defined in tf_convnet.py
It maps the covoltion valriables of the caffe model to the layer in tf model.
Each caffe layer consists of weights and bias  while in tf weights and bias variables are independent
This function works like this only for the original trained caffe model https://lmb.informatik.uni-freiburg.de/resources/opensource/unet/.
To adapt it to different caffe models the names of caffe variables have to be changed
"""

caffe_2_tf_dict = {

    # caffe             # tf weights            #tf bias                shape weights       shape bias
    'conv_d0a-b':       ['down_conv_0/w1:0',    'down_conv_0/b1:0'],    # (3, 3, 1, 64),     (64,)
    'conv_d0b-c':       ['down_conv_0/w2:0',    'down_conv_0/b2:0'],    # (3, 3, 64, 64),    (64,)
    'conv_d1a-b':       ['down_conv_1/w1:0',    'down_conv_1/b1:0'],    # (3, 3, 64, 128),   (128,)
    'conv_d1b-c':       ['down_conv_1/w2:0',    'down_conv_1/b2:0'],    # (3, 3, 128, 128),  (128,)
    'conv_d2a-b':       ['down_conv_2/w1:0',    'down_conv_2/b1:0'],    # (3, 3, 128, 256),  (256,)
    'conv_d2b-c':       ['down_conv_2/w2:0',    'down_conv_2/b2:0'],    # (3, 3, 256, 256),  (256,)
    'conv_d3a-b':       ['down_conv_3/w1:0',    'down_conv_3/b1:0'],    # (3, 3, 256, 512),  (512,)
    'conv_d3b-c':       ['down_conv_3/w2:0',    'down_conv_3/b2:0'],    # (3, 3, 512, 512),  (512,)
    'conv_d4a-b':       ['down_conv_4/w1:0',    'down_conv_4/b1:0'],    # (3, 3, 512, 1024), (1024,)
    'conv_d4b-c':       ['down_conv_4/w2:0',    'down_conv_4/b2:0'],    # (3, 3, 1024, 1024),(1024,)
    'upconv_d4c_u3a':   ['up_conv_3/wd:0',      'up_conv_3/bd:0'],      # (2, 2, 512, 1024), (512,)
    'conv_u3b-c':       ['up_conv_3/w1:0',      'up_conv_3/b1:0'],      # (3, 3, 1024, 512), (512,)
    'conv_u3c-d':       ['up_conv_3/w2:0',      'up_conv_3/b2:0'],      # (3, 3, 512, 512),  (512,)
    'upconv_u3d_u2a':   ['up_conv_2/wd:0',      'up_conv_2/bd:0'],      # (2, 2, 256, 512),  (256,)
    'conv_u2b-c':       ['up_conv_2/w1:0',      'up_conv_2/b1:0'],      # (3, 3, 512, 256),  (256,)
    'conv_u2c-d':       ['up_conv_2/w2:0',      'up_conv_2/b2:0'],      # (3, 3, 256, 256),  (256,),
    'upconv_u2d_u1a':   ['up_conv_1/wd:0',      'up_conv_1/bd:0'],      # (2, 2, 128, 256),  (128,),
    'conv_u1b-c':       ['up_conv_1/w1:0',      'up_conv_1/b1:0'],      # (3, 3, 256, 128),  (128,),
    'conv_u1c-d':       ['up_conv_1/w2:0',      'up_conv_1/b2:0'],      # (3, 3, 128, 128),  (128,),
    'upconv_u1d_u0a':   ['up_conv_0/wd:0',      'up_conv_0/bd:0'],      # (2, 2, 64, 128),   (64,),
    'conv_u0b-c':       ['up_conv_0/w1:0',      'up_conv_0/b1:0'],      # (3, 3, 128, 64),   (64,),
    'conv_u0c-d':       ['up_conv_0/w2:0',      'up_conv_0/b2:0'],      # (3, 3, 64, 64),   (64,),
}

skip_layer = ['conv_u0d-score']


def load_pre_trained_caffe_variables(session, file_path):
    """
    Loads the Varibales (weights and bias) from a given file (hdf5) to the model
    :param file_path: file path of the hdf5 file containing the caffe model
    :param session: tf session
    :return:
    """
    if ".h5" not in file_path:
        raise ValueError()

    print('{} Loading pre-trained caffe weights from {}'.format(datetime.now(), file_path))

    f = h5py.File(file_path, 'r')
    data = f['data']

    for caffe_var, tf_var in caffe_2_tf_dict.items():
        if caffe_var in skip_layer:
            continue
        variables = data[caffe_var]
        # weights
        with tf.variable_scope("", reuse=True):
            # weights
            var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf_var[0])[0]
            # needs to be transposed
            arr = np.array(variables['0']).T
            #  get subarray fram coffe weights. This is necessary for the last Up layer
            # (upconv_u1d_u0a, conv_u0b-c, conv_u0c-d) because unlike the paper they are using 128 instead of 64 kernels
            arr = arr[:var.shape[0], :var.shape[1], :var.shape[2], :var.shape[3]]
            session.run(var.assign(arr))
            # bias
            var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf_var[1])[0]
            arr = np.array(variables['1'])
            arr = arr[:var.shape[0]]
            session.run(var.assign(arr))
