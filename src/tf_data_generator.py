"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.framework.ops import convert_to_tensor
from abc import abstractmethod
import src.utils.tf_utils as tf_utils
import logging
from src.utils.enum_params import TrainingModes, TV_clustering_method
from configuration import DataParams


class TFImageDataGenerator:
    """Wrapper class around the  TensorFlow dataset pipeline.

    Requires TensorFlow >= version 1.12rc0
    """

    def __init__(self, data, data_config, mode):

        self._data_config = data_config  # type: DataParams
        self._in_img_size = self._data_config.raw_image_size
        self._set_img_size = self._data_config.set_image_size
        self._data_max_value = self._data_config.data_max_value
        self._data_norm_value = self._data_config.norm_image_value
        self._raw_data = data
        self._shuffle = self._data_config.shuffle
        self._mode = mode  # type: TrainingModes

        self._nr_of_classes = self._data_config.nr_of_classes
        self._normalize_std = self._data_config.normailze_std
        self._nr_channels = self._data_config.nr_of_image_channels
        self._nr_modalities = self._data_config.nr_of_input_modalities
        self._use_modalities = self._data_config.use_modalities
        self._data_vals = self._data_config.data_values
        self._input_data = None
        self._gt_data = None
        self.data = None
        self.load_data_from_disk = False # not used any more
        self._load_tv_from_file = self._data_config.load_tv_from_file
        self.clustering_method = self._data_config.clustering_method

        self._batch_size = None
        self._buffer_size = None
        self._do_augmentation = None
        self._crop_to_non_zero = None

        self.tv_tau = self._data_config.tv_and_clustering_params["tv_tau"]
        self.tv_weight = self._data_config.tv_and_clustering_params["tv_weight"]
        self.tv_eps = self._data_config.tv_and_clustering_params["tv_eps"]
        self.tv_nr_itr = self._data_config.tv_and_clustering_params["tv_m_itr"]
        self.km_nr_itr = self._data_config.tv_and_clustering_params["km_m_itr"]
        self.mean_shift_n_itr = self._data_config.tv_and_clustering_params["ms_m_itr"] # till convergence
        self.mean_shift_win_size = self._data_config.tv_and_clustering_params["window_size"]
        self.mean_shift_bin_seeding = self._data_config.tv_and_clustering_params["bin_seeding"]
        self.static_cluster_center = self._data_config.tv_and_clustering_params["k_means_pre_cluster"]

        # check if file
        el = next(iter(data)) # nor neccessary any more
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
        in_img = tf.zeros(shape=(self._in_img_size[0], self._in_img_size[1], self._nr_modalities), dtype=tf.float32)
        gt_img = tf.zeros(shape=(self._in_img_size[0], self._in_img_size[1], 1), dtype=tf.float32)
        tv_img = tf.zeros_like(gt_img)
        tv_base = tf.zeros_like(gt_img)

        slices = []
        for i in range(len(self._use_modalities)):
            slices.append(tf_utils.load_png_image(input_ob[i], nr_channels=self._nr_channels,
                                                      img_size=self._in_img_size))
        in_img = tf.concat(slices, axis=2)
        if self._mode == TrainingModes.SEGMENTATION:
            gt_img = tf_utils.load_png_image(gt_ob, nr_channels=self._nr_channels, img_size=self._in_img_size)
        elif self._mode == TrainingModes.TVFLOW_SEGMENTATION or self._mode == TrainingModes.TVFLOW_REGRESSION:
            if self._load_tv_from_file:
                tv_img = tf_utils.load_png_image(gt_ob, nr_channels=self._nr_channels, img_size=self._in_img_size)
            else:
                if None:  # todo add case
                    tv_base = in_img[:, :, i]
                    v = self._data_vals[self._use_modalities[i]]
                    tv_base = tf_utils.normalize_and_zero_center_sclice(tv_base, max=self._data_max_value,
                                                                        new_max=self._data_norm_value,
                                                                        normalize_std=self._normalize_std)
                tv_img = tf_utils.get_tv_smoothed(img=tv_base, tau=self.tv_tau, weight=self.tv_weight,
                                                  eps=self.tv_eps, m_itr=self.tv_nr_itr)

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
        for i in range(len(self._use_modalities)):
            in_img = tf_utils.normalize_and_zero_center_tensor(in_img, modalities=self._use_modalities,
                                                               new_max=self._data_norm_value,
                                                               normalize_std=self._normalize_std,
                                                               data_vals=self._data_vals)
        if self._do_augmentation:
            in_img, gt_img = tf_utils.preprocess_images(in_img, gt_img)

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
        self._batch_size = self._data_config.batch_size_train
        self._buffer_size = self._data_config.buffer_size_train
        self._do_augmentation = self._data_config.do_image_augmentation_train
        self._crop_to_non_zero = self._data_config.crop_to_non_zero_train

        logging.info("Train buffer size {}, batch size {}".format(self._buffer_size, self._batch_size))
        # convert lists to TF tensor
        if self.load_data_from_disk:
            gt = list(self._raw_data.keys())
            input_val = list(self._raw_data.values())
            self._input_data = convert_to_tensor(input_val)
            self._gt_data = convert_to_tensor(gt)
        else:
            self._input_data = np.array([row[1] for row in self._raw_data])
            self._gt_data = np.array([row[0] for row in self._raw_data])

        # create dataset
        tmp_data = tf.data.Dataset.from_tensor_slices((self._input_data, self._gt_data))
        tmp_data = tmp_data.map(self._parse_function, num_parallel_calls=5)
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
        self._batch_size = self._data_config.batch_size_val
        self._buffer_size = self._data_config.buffer_size_val
        self._do_augmentation = self._data_config.do_image_augmentation_val
        self._crop_to_non_zero = self._data_config.crop_to_non_zero_val

        logging.info("Validation buffer size {}, batch size {}".format(self._buffer_size, self._batch_size))
        # convert lists to TF tensor
        if self.load_data_from_disk:
            gt = list(self._raw_data.keys())
            input_val = list(self._raw_data.values())
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
        return self._default_parse_func(input, gt)


