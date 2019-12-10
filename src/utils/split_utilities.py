"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019
"""
import os
import datetime
import json
import logging
from src.utils.enum_params import TrainingModes
from random import shuffle
from configuration import DataParams
import random


class TestFilePaths(object):
    # TODO
    """De-constructor"""

    def __exit__(self, exc_type, exc_value, traceback):
            self.data = None

    """Enter method"""
    def __enter__(self):
        return self

    """Constructor"""
    def __init__(self, paths):
        """ DataPaths object which hold all necessary paths """
        self._paths = paths


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
                 load_test_paths_only=False, new_split=True):
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
        self._data_config = data_config  # type: DataParams
        self._use_scale = self._data_config.use_scale_image_as_gt
        self._load_only_mid_scans = self._data_config.load_only_middle_scans
        self._split_ratio = self._data_config.split_train_val_ratio
        self._nr_of_samples = self._data_config.nr_of_samples
        self._use_mha = self._data_config.use_mha_files_instead
        self._use_modalities = self._data_config.use_modalities
        self._load_tv_from_file = self._data_config.load_tv_from_file
        self.split_name = ""
        self._split_file_extension = '.txt'
        self._base_name = '_split'
        self._tvflow_mode = "tvflow"
        self._seg_mode = "seg"
        self._tvseg_mode ="tv_seg"
        self.validation_paths = dict()
        self.train_paths = dict()
        self.test_paths = dict()

        if self._mode == TrainingModes.TVFLOW_REGRESSION:
            self.split_name = self._tvflow_mode
        elif self._mode == TrainingModes.SEGMENTATION:
            self.split_name = self._seg_mode
        elif self._mode == self._mode == TrainingModes.TVFLOW_SEGMENTATION:
            self.split_name = self._tvseg_mode
        if load_test_paths_only:
            self._read_test_split_only()
            return
        if self._new_split:
            if sum(self._split_ratio) != 1.0:
                raise ValueError()
            self._create_new_split()
            logging.info("Created dataset of {train} training samples and {validation} validation samples.".format(
                train=len(self.train_paths),
                validation=len(self.validation_paths)
            ))
        else:
            self._read_train_and_val_splits_from_folder()
            logging.info("Loaded dataset of {train} training samples and {validation} validation samples from {path}.".format(
                train=len(self.train_paths),
                validation=len(self.validation_paths),
                path=self._paths.split_path
            ))

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
            split = self._get_raw_to_tvflow_file_paths_dict(use_scale=self._use_scale)
            #random.shuffle(split) # Not in use dict.items() is random
            total = self._nr_of_samples
            if self._nr_of_samples == 0:
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

        elif self._mode == TrainingModes.SEGMENTATION or not self._load_tv_from_file:
            train_split, validation_split, test_split = self._get_raw_to_seg_file_paths_dict()
        else:
            raise ValueError("Invalid mode '%s'." % self._mode)

        # safe dataset
        self._safe_and_archive_split(train_split, 'split_{}_train'.format(self.split_name))
        self._safe_and_archive_split(validation_split, 'split_{}_validation'.format(self.split_name))
        self._safe_and_archive_split(test_split, 'split_{}_test'.format(self.split_name))
        self.validation_paths = validation_split
        self.train_paths = train_split
        self.test_paths = test_split

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

        keep_out = []
        if self._load_only_mid_scans:
            keep_out.extend(["_{}.".format(i) for i in range(40)])
            keep_out.extend(["_{}.".format(i) for i in range(120, 150)])

        raw_to_tvflow_file_dict = dict()
        raw_to_tvflow_file_dict.update(self._get_paths_dict_tvflow_single(base_path_key=self._paths.raw_train_dir,
                                                                          base_path_value=self._paths.tv_flow_out_dir,
                                                                          ext_key=self._paths.png_ext,
                                                                          ext_val=tv_flow_ext,
                                                                          gg=self._paths.high_grade_gliomas_folder,
                                                                          without_gt=True,
                                                                          keep_out=keep_out))
        raw_to_tvflow_file_dict.update(self._get_paths_dict_tvflow_single(base_path_key=self._paths.raw_train_dir,
                                                                          base_path_value=self._paths.tv_flow_out_dir,
                                                                          ext_key=self._paths.png_ext,
                                                                          ext_val=tv_flow_ext,
                                                                          gg=self._paths.low_grade_gliomas_folder,
                                                                          without_gt=True,
                                                                          keep_out=keep_out))
        return raw_to_tvflow_file_dict

    def _get_raw_to_seg_file_paths_dict(self):
        """
        Creates a dictionary with segmentation and Brats2015 images with HGG and LGG


        :param use_scale: use scale images from tvflow as gt instead of smoothed images
        :returns Dictionary with input and gt paths:
                {"path/to/brats2015/Patient/Flair/slice.png" : "path/to/brats/Patient/gt/slice.png"}
        """
        if not self._paths.is_loaded:
            return None
        ext = self._paths.png_ext
        directory = self._paths.raw_train_dir
        if self._use_mha:
            directory = self._paths.brats_train_dir
            ext = self._paths.mha_ext

        patient_paths = self._get_patient_folders(base_path=directory,
                                                  gg=self._paths.high_grade_gliomas_folder)
        patient_paths.extend(self._get_patient_folders(base_path=directory,
                                                       gg=self._paths.low_grade_gliomas_folder))

        train_split, validation_split, test_split = self._split_patients(patient_paths=patient_paths)

        keep_out = []
        if self._load_only_mid_scans and not self._use_mha:
            keep_out.extend(["_{}.".format(i) for i in range(40)])
            keep_out.extend(["_{}.".format(i) for i in range(120, 150)])

        train_dict = dict()
        val_dict = dict()
        test_dict = dict()
        train_dict.update(self._get_paths_dict_seg_single(patient_paths=train_split,
                                                          ext_key=ext,
                                                          keep_out=keep_out))
        val_dict.update(self._get_paths_dict_seg_single(patient_paths=validation_split,
                                                          ext_key=ext,
                                                          keep_out=keep_out))
        test_dict.update(self._get_paths_dict_seg_single(patient_paths=test_split,
                                                          ext_key=ext,
                                                          keep_out=keep_out))
        train_k = list(train_dict.keys())
        val_k = list(val_dict.keys())
        random.shuffle(train_k)
        random.shuffle(val_k)
        rand_train_dict = dict()
        for key in train_k:
            rand_train_dict.update({key: train_dict[key]})
        rand_val_dict = dict()
        for key in val_k:
            rand_val_dict.update({key: val_dict[key]})

        return rand_train_dict, rand_val_dict, test_dict

    def _get_patient_folders(self, base_path, gg="HGG"):
        gg_path = os.path.join(base_path, gg)
        patient_names = os.listdir(gg_path)
        patient_paths = [os.path.join(gg_path, patient) for patient in patient_names]
        return  patient_paths

    def _get_paths_dict_tvflow_single(self, base_path_key, base_path_value, ext_key=".png", ext_val=".png",
                                      gg="HGG", without_gt=False, keep_out=[]):
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
        file_dict = dict()
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
                if any(st in file for st in keep_out):
                    continue
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

    def _get_paths_dict_seg_single(self, patient_paths, ext_key=".png", keep_out=[]):
        """
        Creates a dictionary with tvflow and Brats2015


        :param base_path_key: path to brats data
        :param ext_key: file type /ending of brats data
        :param gg: sub path
        :returns Dictionary: with input and gt paths:
                {"path/to/brats2015/Patient/Flair/slice.png" : "path/to/tvflow/Patient/Flair/slice.png"}
        """
        file_dict = dict()
        for patient_path in patient_paths:
            file_paths = os.listdir(patient_path)
            file_paths = sorted(file_paths, reverse=True)
            file_path_gt = [f for f in file_paths if (self._paths.ground_truth_path_identifier[0] in f.lower() or
                                                      self._paths.ground_truth_path_identifier[1] in f.lower())][0]
            for file_path in file_paths:
                if self._paths.ground_truth_path_identifier[0] in file_path.lower() or self._paths.ground_truth_path_identifier[1] in file_path.lower():
                    continue
                file_path_in = file_path
                file_path_full = os.path.join(patient_path, file_path)
                for file in os.listdir(file_path_full):
                    if any(st in file for st in keep_out):
                        continue
                    if file.endswith(ext_key):
                        file_path_key = os.path.join(file_path_full, file)
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
                        file_path_val = file_path_key.replace(file_path_in,
                                                              file_path_gt)
                        if not os.path.exists(file_path_val):
                            continue
                        file_dict[file_path_key] = file_path_val
        return file_dict

    def _safe_and_archive_split(self, split, file_name):
        """
        saves a split as txt and add it to an archive


        :param split: slit to archive
        :param file_name: base file name
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        archive_base_folder = os.path.join(self._paths.split_path, 'archive')
        archive_folder = os.path.join(archive_base_folder, now)
        if not os.path.exists(archive_base_folder):
            os.makedirs(archive_base_folder)
        if not os.path.exists(archive_folder):
            os.makedirs(archive_folder)
        with open(os.path.join(self._paths.split_path, '{}{}'.format(file_name, self._split_file_extension)), 'w') as file:
            file.write(json.dumps(split))
        with open(os.path.join(archive_folder, '{}{}'.format(file_name, self._split_file_extension)), 'w') as file_archive:
            file_archive.write(json.dumps(split))

    def _read_train_and_val_splits_from_folder(self):
        """
        reads a training and validation dataset from txt
        """
        train = self._read_single_split_from_folder(os.path.join(self._paths.split_path,
                                                                 "split_{}_train{}".format(self.split_name,
                                                                                           self._split_file_extension)))
        validation = self._read_single_split_from_folder(
            os.path.join(self._paths.split_path,
                         "split_{}_validation{}".format(self.split_name, self._split_file_extension)))

        self.train_paths = train
        self.validation_paths = validation

    def _read_test_split_only(self):
        """
        reads a test from txt
        """
        test = self._read_single_split_from_folder(os.path.join(self._paths.split_path,
                                                                 "split_{}_test{}".format(self.split_name,
                                                                                           self._split_file_extension)))
        self.test_paths = test

    def _split_patients(self, patient_paths):
        shuffle(patient_paths)
        total = self._nr_of_samples
        if self._nr_of_samples == 0:
            total = len(patient_paths)
        train_split_size = len(patient_paths) * self._split_ratio[0]
        if self._nr_of_samples > train_split_size:
            raise ValueError()

        val_split_size = len(patient_paths) * self._split_ratio[1]
        b_test_split = len(self._split_ratio) == 3
        tsr = 0
        if b_test_split:
            tsr = self._split_ratio[2]
        test_split_size = len(patient_paths) * tsr

        i = 1
        train_split = []
        validation_split = []
        test_split = []
        for p in patient_paths:
            if i <= train_split_size and i <= total:
                train_split.append(p)
            elif i > train_split_size and i <= val_split_size + train_split_size:
                validation_split.append(p)
            elif i <= val_split_size + train_split_size + test_split_size and b_test_split:
                test_split.append(p)
            i += 1

        return train_split, validation_split, test_split

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
