"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from abc import abstractmethod
import src.utils.tf_utils as tf_utils
import logging
from src.utils.enum_params import TrainingModes

class TFImageDataGenerator:
    """Wrapper class around the  TensorFlow dataset pipeline.

    Requires TensorFlow >= version 1.12rc0
    """

    def __init__(self, data, in_img_size, set_img_size, data_max_value=255.0, data_norm_value=1.0, batch_size=32,
                 buffer_size=800, crop_to_non_zero=True, shuffle=False, do_augmentation=False, mode=TrainingModes.TVFLOW_REGRESSION,
                 normalize_std=False, nr_of_classes=1.0, nr_channels=1):

        self._in_img_size = in_img_size
        self._set_img_size = set_img_size
        self._data_max_value = data_max_value
        self._data_norm_value = data_norm_value
        self._raw_data = data
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._shuffle = shuffle
        self._do_augmentation = do_augmentation
        self._mode = mode
        self._crop_to_non_zero = crop_to_non_zero
        self._nr_of_classes = nr_of_classes
        self._normalize_std=normalize_std
        self._nr_channels = nr_channels
        self._input_data = None
        self._gt_data = None
        self.data = None
        self.load_data_from_disk = False
        self.tv_tau = 0.125
        self.tv_weight = 0.2
        self.tv_eps = 0.00001
        self.tv_nr_itr = 100
        self.kmeans_n_itr = 100

        # check if file
        el = next(iter(data))
        if isinstance(el, str):
            self.load_data_from_disk = True

    def __exit__(self, exc_type, exc_value, traceback):
        self._data = None

    def __enter__(self):
        return self

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abstractmethod
    def _parse_function(self):
        raise NotImplementedError()


class TFTrainingImageDataGenerator(TFImageDataGenerator):

    def initialize(self):
        logging.info("Train buffer size {}, batch size {}".format(self._buffer_size, self._batch_size))
        tv_data = convert_to_tensor(1)
        # convert lists to TF tensor
        if self.load_data_from_disk:
            gt = list(self._raw_data.values())
            input_val = list(self._raw_data.keys())
            self._input_data = convert_to_tensor(input_val)
            self._gt_data = convert_to_tensor(gt)
        else:
            self._input_data = np.array([row[1] for row in self._raw_data])
            self._gt_data = np.array([row[0] for row in self._raw_data])

        # create dataset
        tmp_data = tf.data.Dataset.from_tensor_slices((self._input_data, self._gt_data))
        tmp_data = tmp_data.map(self._parse_function, num_parallel_calls=1)
        # shuffle the first `buffer_size` elements of the dataset
        tmp_data = tmp_data.prefetch(buffer_size=self._buffer_size)
        if self._shuffle:
            tmp_data = tmp_data.shuffle(buffer_size=self._buffer_size)
        # create a new dataset with batches of images
        tmp_data = tmp_data.batch(self._batch_size)
        self.data = tmp_data

    def _parse_function(self, input, gt):
        # load and preprocess the image
        # if data is given as png path load the data first
        if self.load_data_from_disk:
            in_img = tf_utils.load_png_image(input, nr_channels=self._nr_channels, img_size=self._in_img_size)
            gt_img = tf_utils.load_png_image(gt, nr_channels=self._nr_channels, img_size=self._in_img_size)
        else:
            in_img = tf.cast(tf.reshape(input, [input.shape[0],  input.shape[1], 1]), tf.float32)
            gt_img = tf.cast(tf.reshape(gt, [gt.shape[0],  gt.shape[1], 1]), tf.float32)

        tv_img = tf.zeros_like(in_img)
        resize_tv = tf.zeros_like(in_img)

        if self._mode == TrainingModes.TVFLOW_REGRESSION:
            gt_img = tf_utils.normalize_and_zero_center_tensor(gt_img,
                                                           max=self._data_max_value,
                                                           new_max=self._data_norm_value,
                                                           normalize_std=self._normalize_std)

        if self._mode == TrainingModes.TVFLOW_SEGMENTATION:
            tv_img, gt_img = tf_utils.get_tv_smoothed_and_kmeans_clusterd_one_hot(image=in_img,
                                                                                  nr_img=self._batch_size,
                                                                                  tv_tau=self.tv_tau,
                                                                                  tv_weight=self.tv_weight,
                                                                                  tv_eps=self.tv_eps,
                                                                                  tv_m_itr=self.tv_nr_itr,
                                                                                  km_cluster_n=self._nr_of_classes,
                                                                                  km_itr_n=self.kmeans_n_itr)

        if self._crop_to_non_zero:
            in_img, gt_img, tv_img = tf_utils.crop_images_to_to_non_zero(scan=in_img, ground_truth=gt_img,
                                                                 size=self._set_img_size, tvimg=tv_img)

        if self._do_augmentation:
            in_img, gt_img = tf_utils.preprocess_images(in_img, gt_img)

        in_img = tf_utils.normalize_and_zero_center_tensor(in_img,
                                                           max=self._data_max_value,
                                                           new_max=self._data_norm_value,
                                                           normalize_std=self._normalize_std)

        if self._mode == TrainingModes.SEGMENTATION:
            gt = tf_utils.to_one_hot(gt_img, depth=self._nr_of_classes)
        elif self._mode == TrainingModes.TVFLOW_SEGMENTATION:
            gt_img = tf.reshape(gt_img, [tf.shape(gt_img)[0], tf.shape(gt_img)[1]])
            gt = tf.one_hot(tf.cast(gt_img, tf.int32), depth=self._nr_of_classes)
            tv_img = tf_utils.normalize_and_zero_center_tensor(tv_img,
                                                               max=self._data_max_value,
                                                               new_max=self._data_norm_value,
                                                               normalize_std=self._normalize_std)

        else:
            gt = gt_img

        if self._set_img_size == self._in_img_size:
            return in_img, gt, tv_img

        resize_in = tf.image.resize_images(in_img, self._set_img_size,
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        resize_gt = tf.image.resize_images(gt, self._set_img_size,
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return resize_in, resize_gt, resize_tv


class TFValidationImageDataGenerator(TFImageDataGenerator):

    def initialize(self):
        logging.info("Validation buffer size {}, batch size {}".format(self._buffer_size, self._batch_size))
        # convert lists to TF tensor
        tv_data = convert_to_tensor(1)
        if self.load_data_from_disk:
            gt = list(self._raw_data.values())
            input_val = list(self._raw_data.keys())
            self._input_data = convert_to_tensor(input_val)
            self._gt_data = convert_to_tensor(gt)
        else:
            self._input_data = np.array([row[1] for row in self._raw_data])
            self._gt_data = np.array([row[0] for row in self._raw_data])

        # create dataset
        tmp_data = tf.data.Dataset.from_tensor_slices((self._input_data, self._gt_data))
        tmp_data = tmp_data.map(self._parse_function, num_parallel_calls=1)
        tmp_data = tmp_data.prefetch(buffer_size=self._buffer_size)
        # shuffle the first `buffer_size` elements of the dataset
        if self._shuffle:
            tmp_data = tmp_data.shuffle(buffer_size=self._buffer_size)
        # create a new dataset with batches of images
        tmp_data = tmp_data.batch(self._batch_size)
        self.data = tmp_data

    def _parse_function(self, input, gt):
        # load and preprocess the image
        # if data is given as png path load the data first

        if self.load_data_from_disk:
            in_img = tf_utils.load_png_image(input, nr_channels=self._nr_channels, img_size=self._in_img_size)
            gt_img = tf_utils.load_png_image(gt, nr_channels=self._nr_channels, img_size=self._in_img_size)
        else:
            in_img = tf.cast(tf.reshape(input, [input.shape[0],  input.shape[1], 1]), tf.float32)
            gt_img = tf.cast(tf.reshape(gt, [gt.shape[0],  gt.shape[1], 1]), tf.float32)

        resize_tv = tf.zeros_like(in_img)
        tv_img = tf.zeros_like(in_img)
        if self._mode == TrainingModes.TVFLOW_REGRESSION:
            gt_img = tf_utils.normalize_and_zero_center_tensor(gt_img,
                                                           max=self._data_max_value,
                                                           new_max=self._data_norm_value,
                                                           normalize_std=self._normalize_std)

        if self._mode == TrainingModes.TVFLOW_SEGMENTATION:
            tv_img, gt_img = tf_utils.get_tv_smoothed_and_kmeans_clusterd_one_hot(image=in_img,
                                                                                  nr_img=self._batch_size,
                                                                                  tv_tau=self.tv_tau,
                                                                                  tv_weight=self.tv_weight,
                                                                                  tv_eps=self.tv_eps,
                                                                                  tv_m_itr=self.tv_nr_itr,
                                                                                  km_cluster_n=self._nr_of_classes,
                                                                                  km_itr_n=self.kmeans_n_itr)

        if self._crop_to_non_zero:
            in_img, gt_img, tv_img = tf_utils.crop_images_to_to_non_zero(scan=in_img, ground_truth=gt_img,
                                                                 size=self._set_img_size, tvimg=tv_img)

        if self._do_augmentation:
            in_img, gt_img = tf_utils.preprocess_images(in_img, gt_img)

        in_img = tf_utils.normalize_and_zero_center_tensor(in_img,
                                                           max=self._data_max_value,
                                                           new_max=self._data_norm_value,
                                                           normalize_std=self._normalize_std)
        if self._mode == TrainingModes.SEGMENTATION:
            gt = tf_utils.to_one_hot(gt_img, depth=self._nr_of_classes)
        elif self._mode == TrainingModes.TVFLOW_SEGMENTATION:
            gt_img = tf.reshape(gt_img, [tf.shape(gt_img)[0], tf.shape(gt_img)[1]])
            gt = tf.one_hot(tf.cast(gt_img, tf.int32), depth=self._nr_of_classes)
            tv_img = tf_utils.normalize_and_zero_center_tensor(tv_img,
                                                               max=self._data_max_value,
                                                               new_max=self._data_norm_value,
                                                               normalize_std=self._normalize_std)

        else:
            gt = gt_img
        if self._set_img_size == self._in_img_size:
            return in_img, gt, tv_img

        resize_in = tf.image.resize_images(in_img, self._set_img_size,
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        resize_gt = tf.image.resize_images(gt, self._set_img_size,
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return resize_in, resize_gt, resize_tv


