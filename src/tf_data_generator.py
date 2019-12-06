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
from src.utils.enum_params import TrainingModes, TV_clustering_method


class TFImageDataGenerator:
    """Wrapper class around the  TensorFlow dataset pipeline.

    Requires TensorFlow >= version 1.12rc0
    """

    def __init__(self, data, in_img_size, set_img_size, data_max_value=255.0, data_norm_value=1.0, batch_size=32,
                 buffer_size=800, crop_to_non_zero=True, shuffle=False, do_augmentation=False, mode=TrainingModes.TVFLOW_REGRESSION,
                 normalize_std=False, nr_of_classes=1.0, nr_channels=1, load_tv_from_file=False,
                 clustering_method=TV_clustering_method.STATIC_BINNING, tv_clustering_params=None):

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
        self._load_tv_from_file = load_tv_from_file
        self.clustering_method = clustering_method

        self.tv_tau = tv_clustering_params.pop("tv_tau", 0.125)
        self.tv_weight = tv_clustering_params.pop("tv_weight", 10000)
        self.tv_eps = tv_clustering_params.pop("tv_eps", 0.00001)
        self.tv_nr_itr = tv_clustering_params.pop("tv_m_itr", 100)
        self.km_nr_itr = tv_clustering_params.pop("km_m_itr", 100)
        self.mean_shift_n_itr = tv_clustering_params.pop("ms_m_itr", -1)  # till convergence
        self.mean_shift_win_size = tv_clustering_params.pop("window_size", 0.01)
        self.mean_shift_bin_seeding = tv_clustering_params.pop("bin_seeding", True)
        self.static_cluster_center = tv_clustering_params.pop("k_means_pre_cluster", [])

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

    def _default_parse_func(self, input_ob, gt_ob):
        # load and preprocess the image
        # if data is given as png path load the data first
        in_img = tf.zeros(self._in_img_size, dtype=tf.float32)
        tv_img = tf.zeros_like(in_img)
        gt_img = tf.zeros_like(in_img)

        if self._mode == TrainingModes.SEGMENTATION:
            if self.load_data_from_disk:
                in_img = tf_utils.load_png_image(input_ob, nr_channels=self._nr_channels, img_size=self._in_img_size)
                gt_img = tf_utils.load_png_image(gt_ob, nr_channels=self._nr_channels, img_size=self._in_img_size)
            else:
                in_img = tf.cast(tf.reshape(input_ob, [input_ob.shape[0], input_ob.shape[1], 1]), tf.float32)
                gt_img = tf.cast(tf.reshape(gt_ob, [gt_ob.shape[0], gt_ob.shape[1], 1]), tf.float32)
        elif self._mode == TrainingModes.TVFLOW_SEGMENTATION or self._mode == TrainingModes.TVFLOW_REGRESSION:
            if self.load_data_from_disk:
                in_img = tf_utils.load_png_image(input_ob, nr_channels=self._nr_channels, img_size=self._in_img_size)
                if self._load_tv_from_file:
                    tv_img = tf_utils.load_png_image(gt_ob, nr_channels=self._nr_channels, img_size=self._in_img_size)
                else:
                    tv_img = tf_utils.get_tv_smoothed(img=in_img, tau=self.tv_tau, weight=self.tv_weight,
                                                      eps=self.tv_eps, m_itr=self.tv_nr_itr)
            else:
                in_img = tf.cast(tf.reshape(input_ob, [input_ob.shape[0], input_ob.shape[1], 1]), tf.float32)
                tv_img = tf.cast(tf.reshape(gt_ob, [gt_ob.shape[0], gt_ob.shape[1], 1]), tf.float32)

            tv_img = tf_utils.normalize_and_zero_center_tensor(tv_img, max=self._data_max_value,
                                                               new_max=self._data_norm_value,
                                                               normalize_std=self._normalize_std)
            if self._mode == TrainingModes.TVFLOW_REGRESSION:
                gt_img = tv_img

            elif self._mode == TrainingModes.TVFLOW_SEGMENTATION:
                if self.clustering_method == TV_clustering_method.STATIC_BINNING:
                    gt_img = tf_utils.get_fixed_bin_clustering(image=tv_img, n_bins=self._nr_of_classes)
                elif self.clustering_method == TV_clustering_method.STATIC_CLUSTERS:
                    gt_img = tf_utils.get_static_clustering(image=tv_img, cluster_centers=self.static_cluster_center)
                elif self.clustering_method == TV_clustering_method.K_MEANS:
                    gt_img = tf_utils.get_kmeans(img=tv_img, clusters_n=self._nr_of_classes, iteration_n=self.km_nr_itr)
                elif self.clustering_method == TV_clustering_method.MEAN_SHIFT:
                    gt_img = tf_utils.get_meanshift_clustering(image=tv_img, ms_itr=self.mean_shift_n_itr,
                                                               win_r=self.mean_shift_win_size,
                                                               n_clusters=self._nr_of_classes,
                                                               bin_seeding=self.mean_shift_bin_seeding)
                else:
                    raise ValueError()
            else:
                raise ValueError()
        else:
            raise ValueError()

        if self._crop_to_non_zero:
            in_img, gt_img, tv_img = tf_utils.crop_images_to_to_non_zero(scan=in_img, ground_truth=gt_img,
                                                                         size=self._set_img_size, tvimg=tv_img)
        if self._do_augmentation:
            in_img, gt_img = tf_utils.preprocess_images(in_img, gt_img)

        in_img = tf_utils.normalize_and_zero_center_tensor(in_img, max=self._data_max_value,
                                                           new_max=self._data_norm_value,
                                                           normalize_std=self._normalize_std)

        if self._mode == TrainingModes.SEGMENTATION:
            gt_img = tf_utils.to_one_hot_custom(gt_img, depth=self._nr_of_classes)

        elif self._mode == TrainingModes.TVFLOW_SEGMENTATION:
            gt_img = tf.reshape(gt_img, [tf.shape(gt_img)[0], tf.shape(gt_img)[1]])
            gt_img = tf.one_hot(tf.cast(gt_img, tf.int32), depth=self._nr_of_classes)

        if self._set_img_size != self._in_img_size:
            in_img = tf.image.resize_images(in_img, self._set_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            gt_img = tf.image.resize_images(gt_img, self._set_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            tv_img = tf.image.resize_images(tv_img, self._set_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return in_img, gt_img, tv_img


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
        tmp_data = tmp_data.map(self._parse_function, num_parallel_calls=3)
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
        return self._default_parse_func(input, gt)


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
        tmp_data = tmp_data.map(self._parse_function, num_parallel_calls=3)
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
        return self._default_parse_func(input, gt)


