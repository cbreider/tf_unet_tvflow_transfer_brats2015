"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
"""

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from abc import ABC, abstractmethod
import da


class ImageDataGenerator(metaclass=ABC):
    """Wrapper class around the  TensorFlow dataset pipeline.

    Requires TensorFlow >= version 1.12rc0
    """
    @abstractmethod
    def __init__(self,):
        raise NotImplementedError()

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

    def __init__(self, file_paths, batch_size=128, buffer_size=8000, shuffle=False, do_pre_processing=False):
        self._file_paths = file_paths
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self._index = 0
        self._shuffle = shuffle
        self._data = None
        self._input_data_paths = None
        self._gt_data_paths = None
        self._do_pre_processing = do_pre_processing

    def initialize(self):
        # convert lists to TF tensor
        self._input_data_paths = convert_to_tensor(self._file_paths.keys(), dtypes=dtypes.string)
        self._gt_data_paths = convert_to_tensor(self._file_paths.keys(), dtypes=dtypes.string)
        # create dataset
        tmp_data = tf.data.Dataset.from_tensor_slices((self._input_data_paths, self._gt_data_paths))
        tmp_data = tmp_data.map(self._parse_function, num_parallel_calls=8)
        # shuffle the first `buffer_size` elements of the dataset
        if self.shuffle:
            tmp_data = tmp_data.shuffle(buffer_size=self.buffer_size)
        # create a new dataset with batches of images
        tmp_data = tmp_data.batch(self.batch_size)
        self._data = tmp_data

    def _parse_function(self, filename_input, filename_gt):
        input_data = dutils.load_2d_volume_as_array(filename=filename_input)
        gt_data = dutils.load_2d_volume_as_array(filename=filename_gt)
        if self._do_pre_processing:
            self._preprocess_files(input_data, gt_data)
        return input_data, gt_data

    def _preprocess_files(self, scan, ground_truth):
        #  preprocess the image
        # TODO
        return scan, ground_truth


class ValidationImageDataGenerator(ImageDataGenerator):

    def __init__(self, file_paths, batch_size=128, buffer_size=8000, shuffle=False):
        self._file_paths = file_paths
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self._index = 0
        self._shuffle = shuffle
        self._data = None
        self._input_data_paths = None
        self._gt_data_paths = None

    def initialize(self):
        # convert lists to TF tensor
        self._input_data_paths = convert_to_tensor(self._file_paths.keys(), dtypes=dtypes.string)
        self._gt_data_paths = convert_to_tensor(self._file_paths.keys(), dtypes=dtypes.string)
        # create dataset
        tmp_data = tf.data.Dataset.from_tensor_slices((self._input_data_paths, self._gt_data_paths))
        tmp_data = tmp_data.map(self._parse_function, num_parallel_calls=8)
        # shuffle the first `buffer_size` elements of the dataset
        if self.shuffle:
            tmp_data = tmp_data.shuffle(buffer_size=self.buffer_size)
        # create a new dataset with batches of images
        tmp_data = tmp_data.batch(self.batch_size)
        self._data = tmp_data

    def _parse_function(self, filename_input, filename_gt):
        input_data = dutils.load_2d_volume_as_array(filename=filename_input)
        gt_data = dutils.load_2d_volume_as_array(filename=filename_gt)
        return input_data, gt_data


class TestImageDataGenerator(ImageDataGenerator):

    def __init__(self, file_paths, batch_size=128, buffer_size=8000, shuffle=False):
        self._file_paths = file_paths
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self._index = 0
        self._shuffle = shuffle
        self._data = None
        self._input_data_paths = None

    def initialize(self):
        # convert lists to TF tensor
        self._input_data_paths = convert_to_tensor(self._file_paths, dtypes=dtypes.string)
        # create dataset
        tmp_data = tf.data.Dataset.from_tensor_slices(self._input_data_paths)
        tmp_data = tmp_data.map(self._parse_function, num_parallel_calls=8)
        # shuffle the first `buffer_size` elements of the dataset
        if self.shuffle:
            tmp_data = tmp_data.shuffle(buffer_size=self.buffer_size)
        # create a new dataset with batches of images
        tmp_data = tmp_data.batch(self.batch_size)
        self._data = tmp_data

    def _parse_function(self, filename):
        input_data = dutils.load_2d_volume_as_array(filename=filename)
        return input_data
