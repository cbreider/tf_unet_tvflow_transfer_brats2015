"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020

"""

import tensorflow as tf
from tensorflow.python.framework.ops import convert_to_tensor
from abc import abstractmethod
import src.utilities.tf_utils as tf_utils
import logging
from src.utilities.enum_params import TrainingModes, TV_clustering_method, Subtumral_Modes
from configuration import Configuration
import math

class TFImageDataGenerator:
    """Wrapper class around the  TensorFlow dataset pipeline.

    Requires TensorFlow >= version 1.12rc0

    """

    def __init__(self, data, data_config, mode):

        self._data_config = data_config  # type: Configuration
        self._in_img_size = self._data_config.raw_image_size
        self._set_img_size = self._data_config.set_image_size
        self._data_max_value = self._data_config.data_max_value
        self._data_norm_value = self._data_config.norm_max_image_value
        self._data_norm_value_tv = self._data_config.norm_max_image_value_tv
        self._raw_data = data
        self._mode = mode  # type: TrainingModes

        self._segmentation_mask = self._data_config.segmentation_mask

        self._nr_of_classes = self._data_config.nr_classes
        self._nr_classes_clustering = self._data_config.nr_clusters
        self._normalize_std = self._data_config.normailze_std
        self._nr_channels = self._data_config.nr_of_image_channels
        self._nr_modalities = self._data_config.nr_input_channels
        self._use_modalities = self._data_config.use_modalities
        self._data_vals = self._data_config.data_values
        self._input_data = None
        self._gt_data = None
        self.data = None
        self._ids = None
        self._load_tv_from_file = False #self._data_config.load_tv_from_file
        self.clustering_method = self._data_config.clustering_method
        self._modalties_tv = self._data_config.use_modalities_for_tv
        self._disort_params = self._data_config.image_disort_params

        self._batch_size = None
        self._buffer_size = None
        self._do_augmentation = None
        self._crop_to_non_zero = None
        self._tv_multi_scale_range = self._data_config.tv_multi_scale_range
        self._tv_static_multi_scale = self._data_config.tv_static_multi_scale
        self.tv_tau = self._data_config.tv_tau
        self.tv_weight = self._data_config.tv_weight
        self.tv_eps = self._data_config.tv_eps
        self.tv_nr_itr = self._data_config.tv_m_itr
        self.km_nr_itr = self._data_config.kmeans_m_iter
        self.mean_shift_n_itr = self._data_config.meanshift_m_iter  # till convergence
        self.mean_shift_win_size = self._data_config.meanshift_window_size
        self.mean_shift_bin_seeding = self._data_config.meanshift_bin_seeding
        self.static_cluster_center = self._data_config.static_cluster_centers

    def __exit__(self, exc_type, exc_value, traceback):
        self._data = None

    def __enter__(self):
        return self

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abstractmethod
    def _parse_function(self, input, gt, values):
        raise NotImplementedError()

    def _default_parse_func(self, input_ob, gt_ob, values):
        """

        :param input_ob: paths of the input scan
        :param gt_ob: path of the ground truth segmentation
        :param values: mean, max and var of the whole underlying input scnas
        :return: normalized in_img, gt_img, tv_img tf tensors
        """
        with tf.name_scope("Parser"):
            # load and preprocess the image
            gt_img = tf.zeros(shape=(self._in_img_size[0], self._in_img_size[1], 1), dtype=tf.float32)
            tv_img = tf.zeros_like(gt_img)

            # load each modality
            slices = []
            for i in range(len(self._use_modalities)):
                slices.append(tf_utils.load_png_image(input_ob[i], nr_channels=self._nr_channels,
                                                          img_size=self._in_img_size))
            in_img = tf.concat(slices, axis=2)  # merge them

            # if BRATS SEGMENTATION mode load ground truth data
            if self._mode == TrainingModes.BRATS_SEGMENTATION or self._mode == TrainingModes.BRATS_SEGMENTATION_TV_PSEUDO_PATIENT:
                gt_img = tf_utils.load_png_image(gt_ob, nr_channels=self._nr_channels, img_size=self._in_img_size)
            # else for tv pre training the gt data will be generated
            elif self._mode == TrainingModes.TVFLOW_SEGMENTATION or self._mode == TrainingModes.TVFLOW_REGRESSION:
                # normalize the input scnas bevore tv smoothing
                norm_std = True if self._mode == TrainingModes.TVFLOW_REGRESSION else False
                if self._modalties_tv:
                    tv_base = tf_utils.normalize_and_zero_center_tensor(in_img, modalities=self._modalties_tv,
                                                                        new_max=self._data_norm_value_tv,
                                                                        normalize_std=True,
                                                                        data_vals=values)

                else:
                    # test wise implementation of combining to modalities.
                    # Seems to not work that well for pre training
                    if self._segmentation_mask == Subtumral_Modes.COMPLETE:  # todo add case
                        tv_base1 = tf.expand_dims(in_img[:, :, 0], axis=2)  # flair + t2
                        tv_base2 = tf.expand_dims(in_img[:, :, 3], axis=2)
                        tv_base1 = tf_utils.normalize_and_zero_center_slice(tv_base1, max=values[0, 0],
                                                                            normalize_std=True,
                                                                            new_max=self._data_norm_value_tv,
                                                                            mean=values[0, 1], var=values[0, 2])
                        tv_base2 = tf_utils.normalize_and_zero_center_slice(tv_base2, max=values[3, 0],
                                                                            normalize_std=True,
                                                                            new_max=self._data_norm_value_tv,
                                                                            mean=values[3, 1], var=values[3, 2])
                        tv_base = (tv_base1 + tv_base2) / 2
                    else:
                        raise ValueError()

                tvs = []
                gts = []
                nr_tv_base = tv_base.get_shape().as_list()[2]
                # run tv smoothing for all modalities to use
                for i in range(nr_tv_base):
                    # if chosen: generate tv smoothed image with a randomly chosen scale
                    if self._tv_multi_scale_range and len(self._tv_multi_scale_range) == 2:
                        tv_weight = tf.random.uniform(minval=self._tv_multi_scale_range[0],
                                                      maxval=self._tv_multi_scale_range[1], shape=())
                        tv = tf_utils.get_tv_smoothed(img=tf.expand_dims(tv_base[:, :, i], axis=2),
                                                            tau=self.tv_tau, weight=tv_weight,
                                                            eps=self.tv_eps, m_itr=self.tv_nr_itr)
                        tvs.append(tv)
                    # if static multi sclae is chosen. generte a tv image for each scale and modality
                    elif self._tv_static_multi_scale:
                        for y in range(len(self._tv_static_multi_scale)):
                            tv_weight = self._tv_static_multi_scale[y]
                            tv = tf_utils.get_tv_smoothed(img=tf.expand_dims(tv_base[:, :, i], axis=2),
                                                                tau=self.tv_tau, weight=tv_weight,
                                                                eps=self.tv_eps, m_itr=self.tv_nr_itr)
                            tvs.append(tv)
                    else:
                        # single scale tv
                        tv_weight = self.tv_weight
                        tv = tf_utils.get_tv_smoothed(img=tf.expand_dims(tv_base[:, :, i], axis=2),
                                                            tau=self.tv_tau, weight=tv_weight, eps=self.tv_eps,
                                                            m_itr=self.tv_nr_itr)
                        tvs.append(tv)


                    # if tv segmentation is chossen cluster tv image depending on the mehod chosen
                    if self._mode == TrainingModes.TVFLOW_SEGMENTATION:
                        if self._data_norm_value_tv:
                            val_range = [-self._data_norm_value_tv, self._data_norm_value_tv]
                        else:
                            no = tf.math.divide((values[i, 0] - values[i, 1]), tf.math.sqrt(values[i, 2]))
                            val_range = [tf.reduce_min(tv), no]

                        if self.clustering_method == TV_clustering_method.STATIC_BINNING:
                            # bin tv image into eqaul fixed bins
                           gts.append(tf_utils.get_fixed_bin_clustering(image=tv, n_bins=self._nr_classes_clustering,
                                                                        val_range=val_range))
                        elif self.clustering_method == TV_clustering_method.STATIC_CLUSTERS:
                            # use pre given cluster centers
                            gts.append(tf_utils.get_static_clustering(image=tv, cluster_centers=self.static_cluster_center))
                        elif self.clustering_method == TV_clustering_method.K_MEANS:
                            # use k-means to cluster image
                            gt = tf_utils.get_kmeans(img=tv, clusters_n=self._nr_of_classes, iteration_n=self.km_nr_itr)
                            gts.append(tf.expand_dims(gt, axis=2))
                        elif self.clustering_method == TV_clustering_method.MEAN_SHIFT:
                            # use mean shift and refitting the number of clusters. ATTENTION: very computational itensive
                            gts.append(tf_utils.get_meanshift_clustering(image=tv, ms_itr=self.mean_shift_n_itr,
                                                                       win_r=self.mean_shift_win_size,
                                                                       n_clusters=self._nr_classes_clustering,
                                                                       bin_seeding=self.mean_shift_bin_seeding))
                # merge all tv output maps
                tv_img = tf.concat(tvs, axis=2)

                if self._mode == TrainingModes.TVFLOW_REGRESSION:
                    gt_img = tv_img
                elif self._mode == TrainingModes.TVFLOW_SEGMENTATION:
                    gt_img = tf.concat(gts, axis=2)


            # normalize the input images
            in_img = tf_utils.normalize_and_zero_center_tensor(in_img, modalities=self._use_modalities,
                                                               new_max=self._data_norm_value,
                                                               normalize_std=self._normalize_std,
                                                               data_vals=values)

            if self._mode == TrainingModes.BRATS_SEGMENTATION_TV_PSEUDO_PATIENT:
                def get_tv_pat(in_pat):
                    tvs = []
                    nr_tv_base = in_pat.get_shape().as_list()[2]
                    # run tv smoothing for all modalities to use
                    for i in range(nr_tv_base):
                        tvs.append(tf_utils.get_tv_smoothed(img=tf.expand_dims(in_pat[:, :, i], axis=2),
                                                            tau=self.tv_tau, weight=self.tv_weight, eps=self.tv_eps,
                                                            m_itr=self.tv_nr_itr))
                    return tf.concat(tvs, axis=2)
                # use tv patient only in 50% of the time
                in_img = tf.cond(tf.greater(tf.random.uniform(shape=(), minval=0.0, maxval=1.0,
                                                              dtype=tf.float32), 0.5),
                                 lambda: in_img,
                                 lambda: get_tv_pat(in_img))

            # crop to non zero area of input image
            if self._crop_to_non_zero:
                in_img, gt_img, tv_img = tf_utils.crop_images_to_to_non_zero(scan=in_img, ground_truth=gt_img,
                                                                             size=self._set_img_size, tvimg=tv_img)
            # do data augmentation
            if self._do_augmentation:
                in_img, gt_img = tf_utils.distort_imgs(in_img, gt_img, params=self._disort_params)

            # if Auto encoder training is chosen the output is just the input
            if self._mode == TrainingModes.AUTO_ENCODER or self._mode == TrainingModes.DENOISING_AUTOENCODER:
                gt_img = in_img
                # if denoising auto encoder is chosen add gaussian noise to the input
                if self._mode == TrainingModes.DENOISING_AUTOENCODER:
                    noise = tf.random_normal(shape=tf.shape(in_img), mean=0.0, stddev=1.0, dtype=tf.float32)
                    mask = tf.cast(tf.greater(tf.random.uniform(shape=tf.shape(in_img), minval=0.0, maxval=1.0,
                                                                dtype=tf.float32), 0.5), tf.float32)
                    in_min = tf.reduce_min(in_img)
                    in_max = tf.reduce_max(in_img)
                    in_img += noise * mask
                    #tf.clip_by_value(in_img, in_min, in_max)

            # genertae one hot tensor for Brats Segmentation
            if self._mode == TrainingModes.BRATS_SEGMENTATION or self._mode == TrainingModes.BRATS_SEGMENTATION_TV_PSEUDO_PATIENT:
                if self._segmentation_mask == Subtumral_Modes.ALL:
                    gt_img = tf.reshape(gt_img, [tf.shape(gt_img)[0], tf.shape(gt_img)[1]])
                    gt_img = tf.one_hot(tf.cast(gt_img, tf.int32), depth=self._nr_of_classes)
                else:
                    gt_img = tf_utils.to_one_hot_brats(gt_img, mask_mode=self._segmentation_mask, depth=self._nr_of_classes)

            # generate one hot tensor for TV segmentation
            elif self._mode == TrainingModes.TVFLOW_SEGMENTATION:
                gt_img = tf.one_hot(tf.cast(gt_img, tf.int32), depth=self._nr_classes_clustering, axis=3)
                gt_img = tf.reshape(gt_img, [tf.shape(gt_img)[0], tf.shape(gt_img)[1], -1])

            if self._set_img_size != self._in_img_size:
                in_img = tf.image.resize_images(in_img, self._set_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                gt_img = tf.image.resize_images(gt_img, self._set_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                tv_img = tf.image.resize_images(tv_img, self._set_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return in_img, gt_img, tv_img, input_ob


class TFTrainingImageDataGenerator(TFImageDataGenerator):

    def initialize(self):
        self._batch_size = self._data_config.batch_size_train
        self._buffer_size = self._data_config.buffer_size_train
        self._do_augmentation = self._data_config.do_image_augmentation_train
        self._crop_to_non_zero = self._data_config.crop_to_non_zero_train

        logging.info("Train buffer size {}, batch size {}".format(self._buffer_size, self._batch_size))
        # convert lists to TF tensor
        gt = list(self._raw_data.keys())
        inputd = list(self._raw_data.values())
        input_imgs = [img[0] for img in inputd]
        input_vals = [v[1] for v in inputd]

        self._input_data = convert_to_tensor(input_imgs)
        self._gt_data = convert_to_tensor(gt)
        self._data_vals = convert_to_tensor(input_vals)
        # create dataset
        tmp_data = tf.data.Dataset.from_tensor_slices((self._input_data, self._gt_data, self._data_vals))
        tmp_data = tmp_data.map(self._parse_function, num_parallel_calls=6)
        # shuffle the first `buffer_size` elements of the dataset
        tmp_data = tmp_data.prefetch(buffer_size=self._buffer_size)
        tmp_data = tmp_data.shuffle(buffer_size=self._buffer_size)
        # create a new dataset with batches of images
        tmp_data = tmp_data.batch(self._batch_size)
        self.data = tmp_data

    def _parse_function(self, input, gt, values):
        # load and preprocess the image
        # if data is given as png path load the data first
        return self._default_parse_func(input, gt, values)


class TFValidationImageDataGenerator(TFImageDataGenerator):
    """ TF Data pipeline generator for Test images without ground truth

     Requires TensorFlow >= version 1.12rc0
     """

    def initialize(self):
        self._batch_size = 1
        self._buffer_size = self._data_config.buffer_size_val
        self._do_augmentation = self._data_config.do_image_augmentation_val
        self._crop_to_non_zero = self._data_config.crop_to_non_zero_val

        if self._mode == TrainingModes.BRATS_SEGMENTATION_TV_PSEUDO_PATIENT:
            self._mode = TrainingModes.BRATS_SEGMENTATION
        logging.info("Validation buffer size {}, batch size {}".format(self._buffer_size, self._batch_size))
        # convert lists to TF tensor
        gt = list(self._raw_data.keys())
        inputd = list(self._raw_data.values())
        input_imgs = [img[0] for img in inputd]
        input_vals = [v[1] for v in inputd]

        self._input_data = convert_to_tensor(input_imgs)
        self._gt_data = convert_to_tensor(gt)
        self._data_vals = convert_to_tensor(input_vals)
        # create dataset
        tmp_data = tf.data.Dataset.from_tensor_slices((self._input_data, self._gt_data, self._data_vals))
        tmp_data = tmp_data.map(self._parse_function, num_parallel_calls=3)
        tmp_data = tmp_data.prefetch(buffer_size=self._buffer_size)
        # shuffle the first `buffer_size` elements of the dataset
        # create a new dataset with batches of images
        tmp_data = tmp_data.batch(self._batch_size)
        self.data = tmp_data

    def _parse_function(self, input, gt, values):
        # load and preprocess the image
        # if data is given as png path load the data first
        return self._default_parse_func(input, gt, values)


class TFTestImageDataGenerator(TFImageDataGenerator):
    """ TF Datapipeline generator for Test images without ground truth
     Requires TensorFlow >= version 1.12rc0
    """
    def initialize(self):
        self._batch_size = 1
        self._buffer_size = self._data_config.buffer_size_val
        self._do_augmentation = False
        self._crop_to_non_zero = False

        logging.info("Test buffer size {}, batch size {}".format(self._buffer_size, self._batch_size))
        # convert lists to TF tensor
        k = list(self._raw_data.keys())

        inputd = list(self._raw_data.values())
        input_imgs = [img[0] for img in inputd]
        input_vals = [v[1] for v in inputd]
        ids = [v[2] for v in inputd]

        self._input_data = convert_to_tensor(input_imgs)
        self._ids = convert_to_tensor(ids)
        self._data_vals = convert_to_tensor(input_vals)
        # create dataset
        tmp_data = tf.data.Dataset.from_tensor_slices((self._input_data, self._data_vals, self._ids))
        tmp_data = tmp_data.map(self._parse_function, num_parallel_calls=3)
        tmp_data = tmp_data.prefetch(buffer_size=self._buffer_size)
        # shuffle the first `buffer_size` elements of the dataset
        # create a new dataset with batches of images
        tmp_data = tmp_data.batch(self._batch_size)
        self.data = tmp_data

    def _parse_function(self, input_ob, values, pat_id):
        """

        :param input_ob: path of the input scans
        :param values: max, mean and var of the complete underlying MRI scan
        :param pat_id: flair ID of the patient. Is is used to store the Scan later on
        :return: normalized in_img, pat_id
        """
        # load and preprocess the image
        # if data is given as png path load the data first
        slices = []
        for i in range(len(self._use_modalities)):
            slices.append(tf_utils.load_png_image(input_ob[i], nr_channels=self._nr_channels,
                                                  img_size=self._in_img_size))
        in_img = tf.concat(slices, axis=2)
        in_img = tf_utils.normalize_and_zero_center_tensor(in_img, modalities=self._use_modalities,
                                                           new_max=self._data_norm_value,
                                                           normalize_std=self._normalize_std,
                                                           data_vals=values)

        return in_img, pat_id


