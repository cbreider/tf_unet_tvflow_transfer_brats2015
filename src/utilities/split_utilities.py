"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""

import os
import datetime
import json
import logging
import numpy as np
from src.utilities.enum_params import TrainingModes
from random import shuffle
from configuration import Configuration
import re
import collections
import src.utilities.io_utils as ioutil
import random
import warnings
from PIL import Image


class TestFilePaths(object):
    # TODO
    """De-constructor"""

    def __exit__(self, exc_type, exc_value, traceback):
            self.data = None

    """Enter method"""
    def __enter__(self):
        return self

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    """Constructor"""
    def __init__(self, paths, patient_paths, config, file_modality="mr_flair", ext=".png"):
        self._paths = paths
        self._use_modalities = config.use_modalities
        self._data_config = config
        self.test_paths = None
        patient_paths.sort(key=self.natural_keys)

        file_dict = collections.OrderedDict()
        for patient_path in patient_paths:
            patient_path = os.path.join(paths.brats_test_dir, patient_path)
            file_paths = os.listdir(patient_path)
            file_paths = sorted(file_paths, reverse=True)
            patient_id = [f for f in file_paths if file_modality in f.lower()][0].split(".")[-1]
            for file_path in file_paths:
                file_path_full = os.path.join(patient_path, file_path)
                file_slices = os.listdir(file_path_full)
                file_slices.sort(key=self.natural_keys)
                modality = ""
                i = 0
                if self._paths.t1_identifier in file_path.lower():
                    modality = self._paths.t1_identifier
                    i = self._use_modalities.index(self._paths.t1_identifier)
                if self._paths.t1c_identifier in file_path.lower():
                    modality = self._paths.t1c_identifier
                    i = self._use_modalities.index(self._paths.t1c_identifier)
                if self._paths.t2_identifier in file_path.lower():
                    modality = self._paths.t2_identifier
                    i = self._use_modalities.index(self._paths.t2_identifier)
                if self._paths.flair_identifier in file_path.lower():
                    modality = self._paths.flair_identifier
                    i = self._use_modalities.index(self._paths.flair_identifier)
                if self._paths.ground_truth_path_identifier[0] in file_path.lower() or \
                        self._paths.ground_truth_path_identifier[1] in file_path.lower():
                    continue

                if not any(modality in m for m in self._use_modalities):
                    continue

                values_path = os.path.join(file_path_full, "values.json")
                if not os.path.exists(values_path):
                    warnings.warn("Values  file for {} not found. Scan will be normailzed by slice value "
                                  "or global values".format(file_path_full), ResourceWarning)

                    mx = self._data_config.data_values[self._use_modalities[i]][0]
                    mn = self._data_config.data_values[self._use_modalities[i]][1]
                    vr = self._data_config.data_values[self._use_modalities[i]][2]

                else:
                    file = open(values_path, 'r')
                    data = file.read()
                    dvals = json.loads(data)
                    mx = dvals["max"]
                    mn = dvals["mean"]
                    vr = dvals["variance"]
                j = 0
                for file in file_slices:
                    if file.endswith(ext):
                        id = "{}_{}".format(patient_id, j)
                        if not id in file_dict:
                            file_dict[id] = [["" for m in range(len(self._use_modalities))],
                                                     [[] for m in range(len(self._use_modalities))],
                                                     patient_id]
                        file_path_img = os.path.join(file_path_full, file)
                        file_dict[id][0][i] = file_path_img
                        file_dict[id][1][i] = [mx, mn, vr]
                    j += 1

        self.test_paths = file_dict


class TrainingDataset(object):

    """TrainingFilePaths Class which can create and save new splits (training and validation file paths)
     or load a previous created one
     """

    """De-constructor"""
    def __exit__(self, exc_type, exc_value, traceback):
        self.data = None

    """Enter method"""
    def __enter__(self):
        return self

    """Constructor"""
    def __init__(self, paths, data_config, mode,
                 load_test_paths_only=False, new_split=True, is_five_fold=False, five_fold_idx=1, nr_of_folds=5,
                 k_fold_nr_val_samples=20):
        """
        Inits a Dataset of training and validation images- Either creates it by reading files from a specific folder
        declared in "paths" or read a existing split form a .txt

        :param paths: instance of DataPaths, which holds all necessary paths
        :param mode: depending on the training mode.
        :param new_split: Flag indicating whether a new split should be created or an old one should be loaded
        :param use_mha: use mha files istead of pngs. Only for SEGMENTATION MODE

        """
        self._paths = paths
        self._new_split = new_split
        self._mode = mode  # type: TrainingModes
        self._data_config = data_config  # type: Configuration
        self._use_scale = self._data_config.use_scale_image_as_gt
        self._load_only_mid_scans = self._data_config.use_only_spatial_range
        self._split_ratio = self._data_config.split_train_val_ratio
        self._nr_training_sample = self._data_config.nr_training_scans
        self._use_modalities = self._data_config.use_modalities
        self._load_tv_from_file = False
        self.split_name = ""
        self._split_file_extension = '.json'
        self._base_name = '_split'
        self._tvflow_mode = "tvflow"
        self._seg_mode = "seg"
        self._tvseg_mode ="tv_seg"
        self._five_fold_folder = "five_folds"
        self._five_fold_file = "fold"
        self._is_five_fold = is_five_fold
        self._five_fold_idx = five_fold_idx - 1
        self.validation_paths = dict()
        self.train_paths = dict()
        self.test_paths = dict()
        self.nr_of_folds = nr_of_folds
        self.k_fold_nr_val_samples = k_fold_nr_val_samples
        self._empyt_slice_ratio = int(1.0 / self._data_config.use_empty_slice_probability)

        logging.info("Loading Dataset...")

        self.split_name = ""

        if load_test_paths_only:
            self._read_test_split_only()
            return
        if self._new_split:
            if sum(self._split_ratio) != 1.0:
                raise ValueError()
            self._create_new_split()
            self.train_paths = self._prune_dict(self.train_paths)

            logging.info("Created dataset of {train} training samples and {validation} validation samples.".format(
                train=len(self.train_paths),
                validation=len(self.validation_paths)
            ))
        else:
            self._read_train_and_val_splits_from_folder()
            self.train_paths = self._prune_dict(self.train_paths)

            logging.info("Loaded dataset of {train} training samples and {validation} validation samples from {path}.".format(
                train=len(self.train_paths),
                validation=len(self.validation_paths),
                path=self._paths.split_path
            ))

    def _prune_dict(self, paths):
        if self._load_only_mid_scans and len(self._load_only_mid_scans) == 2 or self._empyt_slice_ratio:
            tmp = dict()
            keep_out = []
            nt = 0
            t = 0
            idx = 1
            if self._load_only_mid_scans and len(self._load_only_mid_scans) == 2:
                keep_out.extend(["_{}.".format(i) for i in range(0, self._load_only_mid_scans[0])])
                keep_out.extend(["_{}.".format(i) for i in range(self._load_only_mid_scans[1] + 1, 155 + 1)])
            for k in paths.keys():
                keep = False
                if self._load_only_mid_scans and (not any(st in k for st in keep_out)):
                    keep = True
                if self._empyt_slice_ratio:
                    sl = np.array(Image.open(k))
                    mx = np.max(sl)
                    if mx > 0:
                        keep = True
                        t += 1
                    elif mx == 0:
                        idx += 1
                        if idx % self._empyt_slice_ratio == 0:
                            nt += 1
                            keep = True
                if keep:
                    tmp[k] = paths[k]
            paths = tmp

            logging.info("{} non tumor and {} tumor slices".format(nt, t))

        return paths

    def _create_new_split(self):
        """Creates and sets a new split training paths and validtaion paths
            paths are set as dictionary in form of {'path_to_training_file': 'path_to_ground_truth_file'}
        """
        train_split = dict()
        validation_split = dict()
        test_split = dict()
        # mode

        if (self._mode == TrainingModes.TVFLOW_REGRESSION or self._mode == TrainingModes.TVFLOW_SEGMENTATION) \
                and self._load_tv_from_file:
            warnings.warn("_load_tv_from_file is deprecated. Please use built in tv pipeline!", DeprecationWarning)
            split = self._get_raw_to_tvflow_file_paths_dict(use_scale=self._use_scale)
            #random.shuffle(split) # Not in use dict.items() is random
            total = len(split)
            train_split_size = total * self._split_ratio[0]
            val_split_size = total * self._split_ratio[1]
            b_test_split = len(self._split_ratio) == 3
            tsr = 0
            if b_test_split:
                tsr = self._split_ratio[2]
            test_split_size = total * tsr

            i = 0
            for k, v in split.items():
                if i <= train_split_size:
                    train_split[k] = v
                elif i <= val_split_size + train_split_size:
                    validation_split[k] = v
                elif i <= val_split_size + train_split_size + test_split_size and b_test_split:
                    test_split[k] = v
                i += 1
            # safe dataset
            self._safe_and_archive_split(train_split, 'split_{}_train'.format("TV"))
            self._safe_and_archive_split(validation_split, 'split_{}_validation'.format("TV"))
            self._safe_and_archive_split(test_split, 'split_{}_test'.format("TV"))
            self.validation_paths = validation_split
            self.train_paths = train_split
            self.test_paths = test_split

        elif self._mode == TrainingModes.BRATS_SEGMENTATION or not self._load_tv_from_file:
            if self._is_five_fold:
                patients_train, patients_val = self._split_patients_k_fold()
            else:
                patients_train, patients_val = self._split_patients()
            self.validation_paths = self._get_paths_dict_single(patients_val)
            patients_train = self.prune_patients(patients_train)

            logging.info("Loaded {} HGG {} LGG train and {} HGG {} LGG validation scans".format(
                len([i for i in patients_train if "hgg" in i.lower()]),
                len([i for i in patients_train if "lgg" in i.lower()]),
                len([i for i in patients_val if "hgg" in i.lower()]),
                len([i for i in patients_val if "lgg" in i.lower()])
            ))

            train_tmp = self._get_paths_dict_single(patients_train)
            train_k = list(train_tmp.keys())
            random.shuffle(train_k)
            train = dict()
            for key in train_k:
                train.update({key: train_tmp[key]})
            self.train_paths = train

        else:
            raise ValueError("Invalid mode '%s'." % self._mode)

    def _get_patient_folders(self, base_path, gg="HGG"):
        gg_path = os.path.join(base_path, gg)
        patient_names = os.listdir(gg_path)
        patient_paths = [os.path.join(gg_path, patient) for patient in patient_names]
        return patient_paths

    def _get_paths_dict_single(self, patient_paths, ext_key=".png"):
        """
        Creates a dictionary with tvflow and Brats2015


        :param base_path_key: path to brats data
        :param ext_key: file type /ending of brats data
        :param gg: sub path
        :returns Dictionary: with input and gt paths:
                {"path/to/brats2015/Patient/Flair/slice.png" : "path/to/tvflow/Patient/Flair/slice.png"}
        """
        file_dict = collections.OrderedDict()
        val_dict = dict()
        slices = [[] for i in range(len(self._use_modalities) + 1)]
        for patient_path in patient_paths:
            file_paths = os.listdir(patient_path)
            file_paths = sorted(file_paths, reverse=True)
            file_path_gt = [f for f in file_paths if (self._paths.ground_truth_path_identifier[0] in f.lower() or
                                                      self._paths.ground_truth_path_identifier[1] in f.lower())][0]
            for file_path in file_paths:
                file_path_in = file_path
                file_path_full = os.path.join(patient_path, file_path)
                file_slices = os.listdir(file_path_full)
                file_slices.sort(key=self.natural_keys)

                modality = ""
                i = 0
                if self._paths.t1c_identifier in file_path.lower() and self._paths.t1c_identifier in self._use_modalities:
                    modality = self._paths.t1c_identifier
                    i = self._use_modalities.index(self._paths.t1c_identifier)
                elif self._paths.t1_identifier in file_path.lower() and self._paths.t1_identifier in self._use_modalities:
                    modality = self._paths.t1_identifier
                    i = self._use_modalities.index(self._paths.t1_identifier)
                elif self._paths.t2_identifier in file_path.lower() and self._paths.t2_identifier in self._use_modalities:
                    modality = self._paths.t2_identifier
                    i = self._use_modalities.index(self._paths.t2_identifier)
                elif self._paths.flair_identifier in file_path.lower() and self._paths.flair_identifier in self._use_modalities:
                    modality = self._paths.flair_identifier
                    i = self._use_modalities.index(self._paths.flair_identifier)
                elif self._paths.ground_truth_path_identifier[0] in file_path.lower() or \
                        self._paths.ground_truth_path_identifier[1] in file_path.lower():
                    modality = "gt"
                    i = len(self._use_modalities)
                else:
                    continue

                if modality != "gt":
                    values_path = os.path.join(file_path_full, "values.json")
                    if not os.path.exists(values_path):
                        warnings.warn("Values  file for {} not found. Scan will be normailzed by slice value "
                                  "or global values".format(file_path_full),
                                  ResourceWarning)
                        mx = self._data_config.data_values[self._use_modalities[i]][0]
                        mn = self._data_config.data_values[self._use_modalities[i]][1]
                        vr = self._data_config.data_values[self._use_modalities[i]][2]

                    else:
                        file = open(values_path, 'r')
                        data = file.read()
                        dvals = json.loads(data)
                        mx = dvals["max"]
                        mn = dvals["mean"]
                        vr = dvals["variance"]

                for file in file_slices:
                    if file.endswith(ext_key):
                        file_path_img = os.path.join(file_path_full, file)

                        if not any(modality in m for m in self._use_modalities) and modality != "gt":
                            continue

                        file_path_gt_full = file_path_img.replace(file_path_in,
                                                              file_path_gt)
                        if not os.path.exists(file_path_gt_full) and self._mode == TrainingModes.BRATS_SEGMENTATION:
                            warnings.warn("No GT file for {} found. Skipping".format(file_path_img), ResourceWarning)
                            continue

                        slices[i].append(file_path_img)
                        if modality != "gt":
                            val_dict[file_path_img] = [mx, mn, vr]

        # Check if there are the same number of images for each modality
        if any(len(sl) != len(slices[0]) for sl in slices):
            raise ValueError("Length of Lists do not match")
        arr = np.array(slices).transpose()
        for i in range(arr.shape[0]):
            j = arr.shape[1]-1
            values = []
            for x in range(j):
                values.append(val_dict[arr[i, x]])
            file_dict[arr[i, j]] = [list(arr[i, 0:j]), values]
        return file_dict

    def _split_patients(self):

        if not self._paths.is_loaded:
            return None
        ext = self._paths.png_ext
        directory = self._paths.raw_train_dir

        patient_paths_hgg = self._get_patient_folders(base_path=directory,
                                                  gg=self._paths.high_grade_gliomas_folder)
        patient_paths_lgg = self._get_patient_folders(base_path=directory,
                                                      gg=self._paths.low_grade_gliomas_folder)
        shuffle(patient_paths_hgg)
        shuffle(patient_paths_lgg)

        if len(self._split_ratio) == 2:
            self._split_ratio.append(0.0)

        hgg_test_idx = int(self._split_ratio[2] * len(patient_paths_hgg))
        hgg_vali_idx = hgg_test_idx + int(self._split_ratio[1] * len(patient_paths_hgg))

        patient_paths_hgg_test = patient_paths_hgg[:hgg_test_idx]
        patient_paths_hgg_vali = patient_paths_hgg[hgg_test_idx:hgg_vali_idx]
        patient_paths_hgg_train = patient_paths_hgg[hgg_vali_idx:]

        lgg_test_idx = int(self._split_ratio[2] * len(patient_paths_lgg))
        lgg_vali_idx = lgg_test_idx + int(self._split_ratio[1] * len(patient_paths_lgg))

        patient_paths_lgg_test = patient_paths_lgg[:lgg_test_idx]
        patient_paths_lgg_vali = patient_paths_lgg[lgg_test_idx:lgg_vali_idx]
        patient_paths_lgg_train = patient_paths_lgg[lgg_vali_idx:]

        train_combined = patient_paths_hgg_train
        train_combined.extend(patient_paths_lgg_train)
        test_combined = patient_paths_hgg_test
        test_combined.extend(patient_paths_lgg_test)
        vali_combined = patient_paths_hgg_vali
        vali_combined.extend(patient_paths_lgg_vali)

        assert (not any(x in test_combined for x in train_combined))
        assert (not any(x in vali_combined for x in train_combined))
        assert (not any(x in vali_combined for x in test_combined))

        shuffle(train_combined)
        shuffle(test_combined)
        shuffle(vali_combined)

        split = {"training": train_combined, "validation": vali_combined, "testing": test_combined}
        self._safe_and_archive_split(split, "split", is_five_fold=True)

        return train_combined, vali_combined

    def _split_patients_k_fold(self):
        nr_folds = self.nr_of_folds
        folds = []
        if not self._paths.is_loaded:
            return None
        directory = self._paths.raw_train_dir

        patient_paths_hgg = self._get_patient_folders(base_path=directory, gg=self._paths.high_grade_gliomas_folder)
        patient_paths_lgg = self._get_patient_folders(base_path=directory, gg=self._paths.low_grade_gliomas_folder)
        shuffle(patient_paths_hgg)
        shuffle(patient_paths_lgg)

        folds_hgg = ioutil.chunk_list(patient_paths_hgg, nr_folds)
        folds_lgg = ioutil.chunk_list(patient_paths_lgg, nr_folds)
        tmp = []
        for i in range(nr_folds):
            tmp.append(folds_hgg[i] + folds_lgg[nr_folds-1-i])

        for i in range(nr_folds):
            train = [x for j,x in enumerate(tmp) if j!=i]
            train = [y for x in train for y in x]
            val = train[:self.k_fold_nr_val_samples]
            train = train[self.k_fold_nr_val_samples:]
            test = tmp[i]

            assert(not any(x in test for x in train))
            assert(not any(x in val for x in train))
            assert(not any(x in val for x in test))

            shuffle(train)
            shuffle(val)
            shuffle(test)

            folds.append({"training": train, "validation": val, "testing": test})
            self._safe_and_archive_split(folds[i], "{}_{}".format(self._five_fold_file, i), is_five_fold=True)

        pathients_validation = folds[self._five_fold_idx]["validation"]
        patients_train = folds[self._five_fold_idx]["training"]

        return patients_train, pathients_validation

    @staticmethod
    def _read_single_split_from_folder(file_name):
        """
        reads a single dataset from txt

        :param file_name: file path of dataset
        """
        if not os.path.exists(file_name):
            raise ValueError('{0:} file not found'.format(file_name))
        file = open(file_name, 'r')
        data = file.read()
        split = json.loads(data)
        return split

    def prune_patients(self, patients):
        if len(patients) > self._nr_training_sample > 0:
            return patients[:self._nr_training_sample]
        else:
            logging.info("Ivalid Nr of training scans given... Just taking all scans")
            return patients

    def _safe_and_archive_split(self, split, file_name, is_five_fold=False):
        """
        saves a split as txt and add it to an archive

        :param split: slit to archive
        :param file_name: base file name
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        archive_base_folder = os.path.join(self._paths.split_path, 'archive')
        archive_folder = os.path.join(archive_base_folder, "{}{}".format("five_fold" if is_five_fold else "", now))
        if not os.path.exists(archive_base_folder):
            os.makedirs(archive_base_folder)
        if not os.path.exists(archive_folder):
            os.makedirs(archive_folder)
        with open(os.path.join(self._paths.split_path, '{}{}'.format(file_name, self._split_file_extension)), 'w') as file:
            file.write(json.dumps(split))
        with open(os.path.join(self._paths.tf_out_path, '{}{}'.format(file_name, self._split_file_extension)), 'w') as file:
            file.write(json.dumps(split))
        with open(os.path.join(archive_folder, '{}{}'.format(file_name, self._split_file_extension)), 'w') as file_archive:
            file_archive.write(json.dumps(split))

    def _read_train_and_val_splits_from_folder(self):
        """
        reads a training and validation dataset from txt
        """

        if self._is_five_fold:
            fname = os.path.join(self._paths.split_path, "{}_{}{}".format(self._five_fold_file,
                                                                          self._five_fold_idx,
                                                                          self._split_file_extension))
        else:
            fname = os.path.join(self._paths.split_path, "split{}".format(self._split_file_extension))

        split = self._read_single_split_from_folder(fname)

        with open(os.path.join(self._paths.tf_out_path,  fname.split("/")[-1]), 'w') as file:
            file.write(json.dumps(split))

        train = split["training"]
        validation = split["validation"]
        if 1 <= self.k_fold_nr_val_samples < len(validation):
            validation = validation[:self.k_fold_nr_val_samples]

        train = self.prune_patients(train)

        logging.info("Loaded {} HGG {} LGG train and {} HGG {} LGG validation scans".format(
            len([i for i in train if "hgg" in i.lower()]),
            len([i for i in train if "lgg" in i.lower()]),
            len([i for i in validation if "hgg" in i.lower()]),
            len([i for i in validation if "lgg" in i.lower()])))

        train_tmp = self._get_paths_dict_single(train)
        validation = self._get_paths_dict_single(validation)
        train_k = list(train_tmp.keys())
        random.shuffle(train_k)
        train = dict()
        for key in train_k:
            train.update({key: train_tmp[key]})

        self.train_paths = train
        self.validation_paths = validation

    def _read_test_split_only(self):
        """
        reads a test from txt
        """
        if self._is_five_fold:
            test = self._read_single_split_from_folder(os.path.join(self._paths.split_path,
                                                                    "{}_{}{}".format(self._five_fold_file,
                                                                                   self._five_fold_idx,
                                                                                     self._split_file_extension)))["testing"]
            self.test_paths = self._get_paths_dict_single(test)
        else:
            test = self._read_single_split_from_folder(os.path.join(self._paths.split_path,
                                                                 "split{}".format(self._split_file_extension)))["testing"]

        logging.info("Loaded {} HGG {} LGG test".format(
            len([i for i in test if "hgg" in i.lower()]),
            len([i for i in test if "lgg" in i.lower()])))

        self.test_paths = self._get_paths_dict_single(test)

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def _get_raw_to_tvflow_file_paths_dict(self, use_scale=False):
        """
        Creates a dictionary with tvflow and Brats2015 images with HGG and LGG


        :param use_scale: use scale images from tvflow as gt instead of smoothed images
        :returns: Dictionary with input and gt paths:
                {"path/to/brats2015/Patient/Flair/slice.png" : "path/to/tvflow/Patient/Flair/slice.png"}
        """
        if not self._paths.is_loaded:
            return None
        tv_flow_ext = "_tvflow.png"
        if use_scale:
            tv_flow_ext = "_tvflow_scale.png"

        raw_to_tvflow_file_dict = dict()
        raw_to_tvflow_file_dict.update(self._get_paths_dict_tvflow_single(base_path_key=self._paths.raw_train_dir,
                                                                          base_path_value=self._paths.tv_flow_out_dir,
                                                                          ext_key=self._paths.png_ext,
                                                                          ext_val=tv_flow_ext,
                                                                          gg=self._paths.high_grade_gliomas_folder,
                                                                          without_gt=True))
        raw_to_tvflow_file_dict.update(self._get_paths_dict_tvflow_single(base_path_key=self._paths.raw_train_dir,
                                                                          base_path_value=self._paths.tv_flow_out_dir,
                                                                          ext_key=self._paths.png_ext,
                                                                          ext_val=tv_flow_ext,
                                                                          gg=self._paths.low_grade_gliomas_folder,
                                                                          without_gt=True))
        return raw_to_tvflow_file_dict

    def _get_paths_dict_tvflow_single(self, base_path_key, base_path_value, ext_key=".png", ext_val=".png",
                                      gg="HGG", without_gt=False):
        """
        Creates a dictionary with tvflow and Brats2015


        :param base_path_key: path to input data
        :param base_path_value: path to gt data
        :param ext_key: file type /ending of input data
        :param ext_val: file type /ending of gt data
        :param gg: sub path
        :param without_gt: use scale images from tvflow as gt instead of smoothed images
        :returns Dictionary: with input and gt paths:
                {"path/to/brats2015/Patient/Flair/slice.png" : "path/to/tvflow/Patient/Flair/slice.png"}
        """
        file_dict = collections.OrderedDict()
        gg_path = os.path.join(base_path_key, gg)
        patient_names = os.listdir(gg_path)
        patient_paths = [os.path.join(gg_path, patient) for patient in patient_names]
        file_folders = [[os.path.join(path, tmp_path) for tmp_path in os.listdir(path)] for path in patient_paths]
        flat_list_flat = [item for sublist in file_folders for item in sublist]
        for path in flat_list_flat:
            if without_gt and (self._paths.ground_truth_path_identifier[0] in path.lower() or
                               self._paths.ground_truth_path_identifier[1] in path.lower()):
                continue
            out_path = path.replace(base_path_key, base_path_value)
            if not os.path.exists(out_path):
                continue
            for file in os.listdir(path):
                if file.endswith(ext_key):
                    file_path_key = os.path.join(path, file)
                    modality = ""
                    if self._paths.t1_identifier in file_path_key.lower():
                        modality = self._paths.t1_identifier
                    if self._paths.t1c_identifier in file_path_key.lower():
                        modality = self._paths.t1c_identifier
                    if self._paths.t2_identifier in file_path_key.lower():
                        modality = self._paths.t2_identifier
                    if self._paths.flair_identifier in file_path_key.lower():
                        modality = self._paths.flair_identifier
                    if not any(modality in m for m in self._use_modalities):
                        continue
                    file_path_val = file_path_key.replace(base_path_key, base_path_value)
                    file_path_val = file_path_val.replace(ext_key, ext_val)
                    if self._load_tv_from_file and not os.path.exists(file_path_val):
                        continue
                    file_dict[file_path_key] = file_path_val
        return file_dict

