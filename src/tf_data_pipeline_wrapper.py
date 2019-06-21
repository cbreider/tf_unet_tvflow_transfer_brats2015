"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019
"""

from src.data_generator import TrainingImageDataGenerator, ValidationImageDataGenerator, TestImageDataGenerator
import src.configuration as config
import tensorflow as tf
import tensorflow.data as tf_data


class ImageData(object):
    """Wrapper class around the Tensorflow dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, file_paths, batch_size=128, buffer_size=800, shuffle=False, do_pre_processing=False,
                 mode=config.DataModes.TRAINING):
        """
        Inits a Tensorflow data pipeline

        :param file_paths: dictionary of paths to files { "paths/to/input_file" : "path/to/gt_file"}
        :param batch_size: size of each batch returned by next_batch
        :param buffer_size: number of samples to be prefetched in buffer to increase learing speed (need RAM)
        :param shuffle: shuffle data set
        :param do_pre_processing: pre process images see tf_utils.preprocess_images. Only used in Traing mode
        :param mode: mode either training,validtation or testing
        """
        self._data_paths = file_paths
        self._batch_size = batch_size
        self._mode = mode
        self._buffer_size = buffer_size
        self._shuffle = shuffle
        self._do_pre_processing = do_pre_processing
        self.data_generator = None
        self.init_op = None
        self.next_batch = None
        if not (mode == config.DataModes.TRAINING or
                mode == config.DataModes.VALIDATION or
                mode == config.DataModes.TESTING):
            raise ValueError("Invalid mode {}".format(self._mode))

    def __exit__(self, exc_type, exc_value, traceback):
        self._data_generator = None

    def __enter__(self):
        return self

    def create(self):
        """
        creates a Tensorflow data pipeline based on a data generator
        """
        # load and preprocess the data on the cpu
        with tf.device('/cpu:0'):
            graph = tf.get_default_graph()
            with graph.as_default():
                if self._mode == config.DataModes.TRAINING:
                    self.data_generator = TrainingImageDataGenerator(file_paths=self._data_paths,
                                                                     batch_size=self._batch_size,
                                                                     buffer_size=self._buffer_size,
                                                                     shuffle=self._shuffle,
                                                                     do_pre_processing=self._do_pre_processing)
                elif self._mode == config.DataModes.VALIDATION:
                    self.data_generator = ValidationImageDataGenerator(file_paths=self._data_paths,
                                                                       batch_size=self._batch_size,
                                                                       buffer_size=self._buffer_size,
                                                                       shuffle=self._shuffle,
                                                                       do_pre_processing=self._do_pre_processing)
                elif self._mode == config.DataModes.TESTING:
                    self.data_generator = TestImageDataGenerator(file_paths=self._data_paths,
                                                                 batch_size=self._batch_size,
                                                                 buffer_size=self._buffer_size,
                                                                 shuffle=self._shuffle,
                                                                 do_pre_processing=self._do_pre_processing)
                else:
                    raise ValueError("Invalid mode {}".format(self._mode))
                self.data_generator.initialize()
                # create an reinitializable iterator given the dataset structure
                iterator = tf_data.Iterator.from_structure(self.data_generator.data.output_types,
                                                           self.data_generator.data.output_shapes)

                next_batch = iterator.get_next()
            # Ops for initializing the two different iterators
        self.init_op = iterator.make_initializer(self.data_generator.data)
        self.next_batch = next_batch
