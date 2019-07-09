"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019
"""

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from abc import ABC, abstractmethod
import src.utils.tf_utils as tf_utils
import src.configuration as config
import logging


class ImageDataGenerator:
    """Wrapper class around the  TensorFlow dataset pipeline.

    Requires TensorFlow >= version 1.12rc0
    """

    @abstractmethod
    def __init__(self):
        return

    def __exit__(self, exc_type, exc_value, traceback):
        self.data = None

    def __enter__(self):
        return self

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abstractmethod
    def _parse_function(self):
        raise NotImplementedError()


class TrainingImageDataGenerator(ImageDataGenerator):

    def __init__(self, file_paths, batch_size=32, buffer_size=800, shuffle=False, do_pre_processing=False,
                 mode=config.TrainingModes.TVFLOW):
        self._file_paths = file_paths
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self._index = 0
        self._shuffle = shuffle
        self.data = None
        self._input_data_paths = None
        self._gt_data_paths = None
        self._do_pre_processing = do_pre_processing
        self.mode = mode

    def initialize(self):
        logging.info("Train buffer size {}, batch size {}".format(self.buffer_size, self.batch_size))

        # convert lists to TF tensor
        input_val = list(self._file_paths.keys())
        gt = list(self._file_paths.values())
        self._input_data_paths = convert_to_tensor(input_val, dtype=dtypes.string)
        self._gt_data_paths = convert_to_tensor(gt, dtype=dtypes.string)
        # create dataset
        tmp_data = tf.data.Dataset.from_tensor_slices((self._input_data_paths, self._gt_data_paths))
        tmp_data = tmp_data.map(self._parse_function, num_parallel_calls=8)
        # shuffle the first `buffer_size` elements of the dataset
        tmp_data = tmp_data.prefetch(buffer_size=self.buffer_size)
        if self._shuffle:
            tmp_data = tmp_data.shuffle(buffer_size=self.buffer_size)
        # create a new dataset with batches of images
        tmp_data = tmp_data.batch(self.batch_size)
        self.data = tmp_data

    def _parse_function(self, filename_input, filename_gt):
        # load and preprocess the image
        in_img = tf_utils.load_png_image(filename_input)
        gt_img = tf_utils.load_png_image(filename_gt)
        if self._do_pre_processing:
            in_img, gt_img = tf_utils.preprocess_images(in_img, gt_img)
        if self.mode == config.TrainingModes.TVFLOW:
            gt_one_hot = tf_utils.convert_8bit_image_to_one_hot(gt_img, depth=config.DataParams.nr_of_classes_tv_flow_mode)
        elif self.mode == config.TrainingModes.SEGMENTATION:
            gt_one_hot = tf_utils.convert_8bit_image_to_one_hot(gt_img, depth=config.DataParams.nr_of_classes_seg_mode)
        else:
            raise ValueError()
        return in_img, gt_one_hot


class ValidationImageDataGenerator(ImageDataGenerator):

    def __init__(self, file_paths, batch_size=32, buffer_size=800, shuffle=False, do_pre_processing=False,
                 mode=config.TrainingModes.TVFLOW):
        self._file_paths = file_paths
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self._index = 0
        self._shuffle = shuffle
        self.data = None
        self._input_data_paths = None
        self._gt_data_paths = None
        self._do_pre_processing = do_pre_processing
        self.mode = mode

    def initialize(self):
        logging.info("Validation buffer size {}, batch size {}".format(self.buffer_size, self.batch_size))
        # convert lists to TF tensor
        input_val = list(self._file_paths.keys())
        gt = list(self._file_paths.values())
        self._input_data_paths = convert_to_tensor(input_val, dtype=dtypes.string)
        self._gt_data_paths = convert_to_tensor(gt, dtype=dtypes.string)
        # create dataset
        tmp_data = tf.data.Dataset.from_tensor_slices((self._input_data_paths, self._gt_data_paths))
        tmp_data = tmp_data.map(self._parse_function, num_parallel_calls=8)
        tmp_data = tmp_data.prefetch(buffer_size=self.buffer_size)
        # shuffle the first `buffer_size` elements of the dataset
        if self._shuffle:
            tmp_data = tmp_data.shuffle(buffer_size=self.buffer_size)
        # create a new dataset with batches of images
        tmp_data = tmp_data.batch(self.batch_size)
        self.data = tmp_data

    def _parse_function(self, filename_input, filename_gt):
        # load and preprocess the image
        in_img = tf_utils.load_png_image(filename_input)
        gt_img = tf_utils.load_png_image(filename_gt)
        if self.mode == config.TrainingModes.TVFLOW:
            gt_one_hot = tf_utils.convert_8bit_image_to_one_hot(gt_img, depth=config.DataParams.nr_of_classes_tv_flow_mode)
        elif self.mode == config.TrainingModes.SEGMENTATION:
            gt_one_hot = tf_utils.convert_8bit_image_to_one_hot(gt_img, depth=config.DataParams.nr_of_classes_seg_mode)
        else:
            raise ValueError()
        return in_img, gt_one_hot


class TestImageDataGenerator(ImageDataGenerator):

    def __init__(self, file_paths, batch_size=128, buffer_size=8000, shuffle=False, do_pre_processing=False,
                 mode=config.TrainingModes.TVFLOW):
        self._file_paths = file_paths
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self._index = 0
        self._shuffle = shuffle
        self.data = None
        self._input_data_paths = None
        self._gt_data_paths = None
        self._do_pre_processing = do_pre_processing
        self.mode = mode

    def initialize(self):
        # convert lists to TF tensor
        self._input_data_paths = convert_to_tensor(self._file_paths, dtype=dtypes.string)
        # create dataset
        tmp_data = tf.data.Dataset.from_tensor_slices(self._input_data_paths)
        tmp_data = tmp_data.map_fn(self._parse_function, num_parallel_calls=8)
        # shuffle the first `buffer_size` elements of the dataset
        if self._shuffle:
            tmp_data = tmp_data.shuffle(buffer_size=self.buffer_size)
        # create a new dataset with batches of images
        tmp_data = tmp_data.batch(self.batch_size)
        self.data = tmp_data

    def _parse_function(self, filename):
        # load and preprocess the image
        in_img = tf_utils.load_png_image(filename)
        return in_img
