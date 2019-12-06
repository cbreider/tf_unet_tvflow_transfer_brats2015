"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019
"""

from src.tf_data_generator import TFTrainingImageDataGenerator, TFValidationImageDataGenerator
import tensorflow as tf
import tensorflow.data as tf_data
from src.utils.enum_params import TrainingModes, DataModes, TV_clustering_method


class ImageData(object):
    """Wrapper class around the Tensorflow dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, data, data_config, mode=DataModes.TRAINING, train_mode=TrainingModes.TVFLOW_REGRESSION):
        """
        Inits a Tensorflow data pipeline

        :param data: dictionary of paths to files { "paths/to/input_file" : "path/to/gt_file"} or dict of data
        :param batch_size: size of each batch returned by next_batch
        :param buffer_size: number of samples to be prefetched in buffer to increase learing speed (need RAM)
        :param shuffle: shuffle data set
        :param do_img_augmentation: pre process images see tf_utils.preprocess_images. Only used in Traing mode
        :param mode: mode either training,validtation or testing
        """
        self._data = data
        self._mode = mode
        self._data_config = data_config
        self.data_generator = None
        self.init_op = None
        self.next_batch = None
        self.train_mode = train_mode
        self.size = len(self._data)
        if not (mode == DataModes.TRAINING or
                mode == DataModes.VALIDATION or
                mode == DataModes.TESTING):
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
                if self._mode == DataModes.TRAINING:
                    self.data_generator = TFTrainingImageDataGenerator(data=self._data,
                                                                       mode=self.train_mode,
                                                                       data_config=self._data_config)

                elif self._mode == DataModes.VALIDATION:
                    self.data_generator = TFValidationImageDataGenerator(data=self._data,
                                                                         mode=self.train_mode,
                                                                         data_config=self._data_config)
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
