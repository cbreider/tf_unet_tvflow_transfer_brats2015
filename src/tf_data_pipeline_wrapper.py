"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019
"""

from src.tf_data_generator import TFTrainingImageDataGenerator, TFValidationImageDataGenerator, TFTestImageDataGenerator
import tensorflow as tf
import tensorflow.data as tf_data
from src.utils.enum_params import TrainingModes, DataModes

class ImageData(object):
    """Wrapper class around the Tensorflow dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, file_paths, in_img_size, set_img_size, data_max_value=255.0, data_norm_value=1.0,
                 crop_to_non_zero=False, do_augmentation=False, normalize_std=False, nr_of_classes=1.0,
                 nr_channels=1, batch_size=128, buffer_size=800, shuffle=False,
                 mode=DataModes.TRAINING, train_mode=TrainingModes.TVFLOW):
        """
        Inits a Tensorflow data pipeline

        :param file_paths: dictionary of paths to files { "paths/to/input_file" : "path/to/gt_file"}
        :param batch_size: size of each batch returned by next_batch
        :param buffer_size: number of samples to be prefetched in buffer to increase learing speed (need RAM)
        :param shuffle: shuffle data set
        :param do_img_augmentation: pre process images see tf_utils.preprocess_images. Only used in Traing mode
        :param mode: mode either training,validtation or testing
        """
        self._data_paths = file_paths
        self._batch_size = batch_size
        self._mode = mode
        self._buffer_size = buffer_size
        self._shuffle = shuffle
        self._in_img_size = in_img_size
        self._set_img_size = set_img_size
        self._data_max_value = data_max_value
        self._data_norm_value = data_norm_value,
        self._crop_to_non_zero = crop_to_non_zero,
        self._do_augmentation = do_augmentation,
        self._normalize_std = normalize_std,
        self._nr_of_classes = nr_of_classes,
        self._nr_channels = nr_channels
        self.data_generator = None
        self.init_op = None
        self.next_batch = None
        self.train_mode = train_mode
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
                    self.data_generator = TFTrainingImageDataGenerator(file_paths=self._data_paths,
                                                                       batch_size=self._batch_size,
                                                                       buffer_size=self._buffer_size,
                                                                       shuffle=self._shuffle,
                                                                       mode=self.train_mode,
                                                                       in_img_size=self._in_img_size,
                                                                       set_img_size=self._set_img_size,
                                                                       data_max_value=self._data_max_value,
                                                                       data_norm_value=self._data_norm_value,
                                                                       crop_to_non_zero=self._crop_to_non_zero,
                                                                       do_augmentation=self._do_augmentation,
                                                                       normalize_std=self._normalize_std,
                                                                       nr_of_classes=self._nr_of_classes,
                                                                       nr_channels=self._nr_channels)

                elif self._mode == DataModes.VALIDATION:
                    self.data_generator = TFValidationImageDataGenerator(file_paths=self._data_paths,
                                                                         batch_size=self._batch_size,
                                                                         buffer_size=self._buffer_size,
                                                                         shuffle=self._shuffle,
                                                                         mode=self.train_mode,
                                                                         in_img_size=self._in_img_size,
                                                                         set_img_size=self._set_img_size,
                                                                         data_max_value=self._data_max_value,
                                                                         data_norm_value=self._data_norm_value,
                                                                         crop_to_non_zero=self._crop_to_non_zero,
                                                                         do_augmentation=self._do_augmentation,
                                                                         normalize_std=self._normalize_std,
                                                                         nr_of_classes=self._nr_of_classes,
                                                                         nr_channels=self._nr_channels)
                elif self._mode == DataModes.TESTING:
                    self.data_generator = TFTestImageDataGenerator(file_paths=self._data_paths,
                                                                   batch_size=self._batch_size,
                                                                   buffer_size=self._buffer_size,
                                                                   shuffle=self._shuffle,
                                                                   in_img_size=self._in_img_size,
                                                                   set_img_size=self._set_img_size,
                                                                   data_max_value=self._data_max_value,
                                                                   data_norm_value=self._data_norm_value,
                                                                   normalize_std=self._normalize_std,
                                                                   nr_of_classes=self._nr_of_classes,
                                                                   nr_channels=self._nr_channels)
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
