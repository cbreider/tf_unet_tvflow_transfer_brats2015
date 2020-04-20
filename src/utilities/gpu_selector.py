"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""


import os
import logging

cuda_device = ''


def set_cuda_gpu(cuda_gpu):
    """
    Selects a CUDA device (GPU) with the given index. It does so by setting the GPU to the only visible CUDA device
    for the OS

    :param cuda_gpu: index of the CUDA device to select
    """
    global cuda_device
    cuda_device = str(cuda_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    logging.info('Setting Cuda GPU /{}'.format(cuda_device))

    # calling this function makes tf allocating most of gpu memory. Why? Is this bad?
    # local_device_protos = device_lib.list_local_devices()
    # print([x.name for x in local_device_protos])


def reset():
    """
    Resets the CUDA visible devices OS parameter
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
