"""
Author: Christian Breiderhoff
"""

import tensorflow as tf
from src.utils.data_generator import TrainingImageDataGenerator, ValidationImageDataGenerator, TestImageDataGenerator
import tensorflow.data as tf_data


class ImageData(object):
    """Wrapper class around the  Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, batch_size, mode="training"):

        self.batch_size = batch_size
        self._data_generator = None
        self._mode = mode
        self._data_paths = None
        self._dataGenerator = None
        self.init_op = None
        self.next_batch = None
        if not (mode == "training" or mode == "validation" or mode == "test"):
            raise ValueError("Invalid mode {}".format(self._mode))

    def __exit__(self, exc_type, exc_value, traceback):
        self._data_generator = None

    def __enter__(self):
        return self

    def create(self, file_paths):
        # load and preprocess the data on the cpu
        # with tf.device('/cpu:0'):
        self._data_paths = file_paths
        if self._mode == 'training':
            self._data_generator = TrainingImageDataGenerator(file_paths=self._data_paths,
                                                              batch_size=128,
                                                              buffer_size=8000, shuffle=True,
                                                              do_pre_processing=True)
        elif self._mode == 'validation':
            self._data_generator = ValidationImageDataGenerator(file_paths=self._data_paths,
                                                                batch_size=128,
                                                                buffer_size=8000, shuffle=True)
        elif self._mode == 'test':
            self._data_generator = TrainingImageDataGenerator(file_paths=self._data_paths,
                                                              batch_size=128,
                                                              buffer_size=8000, shuffle=True)
        else:
            raise ValueError("Invalid mode {}".format(self._mode))
        # create an reinitializable iterator given the dataset structure
        iterator = tf_data.Iterator.from_structure(self._data_generator.data.output_types,
                                                   self._data_generator.data.output_shapes)
        next_batch = iterator.get_next()
        # Ops for initializing the two different iterators
        self.init_op = iterator.make_initializer(self.test_data_paths.data)
        self.next_batch = next_batch
