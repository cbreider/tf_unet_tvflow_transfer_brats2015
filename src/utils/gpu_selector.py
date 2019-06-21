"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
"""


import os
from tensorflow.python.client import device_lib

cuda_device = ''


def set_cuda_gpu(cuda_gpu):

    global cuda_device
    cuda_device = cuda_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    print('Setting Cuda GPU /{}'.format(cuda_device))

    # calling this function makes tf allocating most of gpu memory. Why? Is this bad?
    # local_device_protos = device_lib.list_local_devices()
    # print([x.name for x in local_device_protos])


def is_gpu_availbale():

    # HACK HACK HACK
    return cuda_device != ''

    # local_device_protos = device_lib.list_local_devices()
    # return len(local_device_protos) > 1


def reset():

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
