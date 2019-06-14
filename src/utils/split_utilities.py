"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
"""

import os
import datetime
from src.utils.path_utils import DataPaths


class TestFilePaths(object):
    class TrainingFilePaths(object):
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
            self.initialze()


class TrainingFilePaths(object):

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
    def __init__(self, paths, mode="tvflow_training", new_split=True, split_ratio=0.7, nr_of_samples=0):
        """ DataPaths object which hold all necessary paths """
        self._paths = paths
        """boolean which holds wether a new split should be created or ar saved split should be read"""
        self._new_split = new_split         #
        """mode could have two values (string) 
            1. "tvflow_training": indicates that cnn should be pre-trained in tvflow smoothed images
            2. "seg_training": indicates that cnn should be trained with segmetation maps """
        self._mode = mode
        self._split_ratio = split_ratio     # flaoting point number
        self._nr_of_samples = nr_of_samples
        self.split_name = ""
        self._file_extension = '.txt'
        self._base_name = '_split'
        self._tvflow_mode = "tvflow"
        self._seg_mode = "seg"
        self.validation_paths = None
        self.train_paths = None
        if self._new_split:
            self._create_new_split()
        else:
            self._read_train_and_val_splits_from_folder()

    """Creates and sets a new split training paths and validtaion paths
        paths are set as dictionary in form of {'path_to_training_file': 'path_to_ground_truth_file'}
    """
    def _create_new_split(self):
        # mode
        if self._mode == "tvflow_training":
            split = self._get_raw_to_tvflow_file_paths_dict(use_scale=False)
            self.split_name = self._tvflow_mode
        elif self._mode == "seg_training":
            self.split_name = self._seg_mode
            split = self._get_raw_to_seg_file_paths_dict()
        else:
            raise ValueError("Invalid mode '%s'." % self._mode)
        # random.shuffle(split) # Not in use dict.items() is random
        total = len(split)
        train_split_size = total * self._split_ratio
        train_split = []
        validation_split = []
        for i in range(total):
            if i <= train_split_size:
                train_split.append(split[i])
            else:
                validation_split.append(split[i])
        self._safe_and_archive_split(train_split, 'split_{}_train'.format(self.split_name))
        self._safe_and_archive_split(validation_split, 'split_{}_validation'.format(self.split_name))
        self.validation_paths = validation_split
        self.train_paths = train_split

    def _get_raw_to_tvflow_file_paths_dict(self, use_scale=False):

        if not self._paths.is_loaded:
            return None
        tv_flow_ext = "_tvflow.nrrd"
        if use_scale:
            tv_flow_ext = "_tvflow_scale.nrrd"

        raw_to_tvflow_file_dict = dict()
        raw_to_tvflow_file_dict.update(self._get_paths_dict_tvflow_single(base_path_key=self._paths.raw_train_dir,
                                                                          base_path_value=self._paths.tv_flow_out_dir,
                                                                          ext_key=self._paths.in_ext,
                                                                          ext_val=tv_flow_ext,
                                                                          gg=self._paths.high_grade_gliomas_folder,
                                                                          without_gt=True))
        raw_to_tvflow_file_dict.update(self._get_paths_dict_tvflow_single(base_path_key=self._paths.raw_train_dir,
                                                                          base_path_value=self._paths.tv_flow_out_dir,
                                                                          ext_key=self._paths.in_ext,
                                                                          ext_val=tv_flow_ext,
                                                                          gg=self._paths.low_grade_gliomas_folder,
                                                                          without_gt=True))

        return raw_to_tvflow_file_dict

    def _get_raw_to_seg_file_paths_dict(self):

        if not self._paths.is_loaded:
            return None

        raw_to_seg_file_dict = dict()
        raw_to_seg_file_dict.update(self._get_paths_dict_seg_single(base_path=self._paths.raw_train_dir,
                                                                    ext_key=self._paths.in_ext,
                                                                    gg=self._paths.high_grade_gliomas_folder))
        raw_to_seg_file_dict.update(self._get_paths_dict_seg_single(base_path=self._paths.raw_train_dir,
                                                                    ext_key=self._paths.in_ext,
                                                                    gg=self._paths.low_grade_gliomas_folder))

        return raw_to_seg_file_dict

    def _get_paths_dict_tvflow_single(self, base_path_key, base_path_value, ext_key=".nrrd", ext_val=".nrrd",
                                      gg="HGG", without_gt=False):
        file_dict = dict()
        gg_path = os.path.join(base_path_key, gg)
        patient_names = os.listdir(gg_path)
        patient_paths = [os.path.join(gg_path, patient) for patient in patient_names]
        file_folders = [[os.path.join(path, tmp_path) for tmp_path in os.listdir(path)] for path in patient_paths]
        flat_list_flat = [item for sublist in file_folders for item in sublist]
        for path in flat_list_flat:
            if without_gt and self._paths.ground_truth_path_identifier in path.lower():
                continue
            out_path = path.replace(base_path_key, base_path_value)
            if not os.path.exists(out_path):
                raise FileNotFoundError()
            for file in os.listdir(path):
                if file.endswith(ext_key):
                    file_path_key = os.path.join(path, file)
                    file_path_val = file_path_key.replace(base_path_key, base_path_value)
                    file_path_val = file_path_val.replace(ext_key, ext_val)
                    if not os.path.exists(file_path_val):
                        raise FileNotFoundError()
                    file_dict[file_path_key] = file_path_val
        return file_dict

    def _get_paths_dict_seg_single(self, base_path, ext_key=".nrrd", gg="HGG",):
        file_dict = dict()
        gg_path = os.path.join(base_path, gg)
        patient_names = os.listdir(gg_path)
        patient_paths = [os.path.join(gg_path, patient) for patient in patient_names]
        for patient_path in patient_paths:
            file_paths = [os.path.join(patient_path, path) for path in os.listdir(patient_path)]
            for file_path in file_paths:
                if self._paths.ground_truth_path_identifier in file_path.lower():
                    continue
                for file in os.listdir(file_path):
                    if file.endswith(ext_key):
                        file_path_key = os.path.join(file_path, file)
                        modality = ""
                        if self._paths.t1_identifier in file_path_key.lower():
                            modality = self._paths.t1_identifier
                        if self._paths.t1c_identifier in file_path_key.lower():
                            modality = self._paths.t1c_identifier
                        if self._paths.t2_identifier in file_path_key.lower():
                            modality = self._paths.t2_identifier
                        if self._paths.flair_identifier in file_path_key.lower():
                            modality = self._paths.flair_identifier
                        file_path_val = file_path_key.replace(modality, self._paths.ground_truth_path_identifier)
                        if not os.path.exists(file_path_val):
                            raise FileNotFoundError()
                        file_dict[file_path_key] = file_path_val
        return file_dict

    def _safe_and_archive_split(self, split, file_name):
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        archive_base_folder = os.path.join(self._paths.split_path, 'archive')
        archive_folder = os.path.join(archive_base_folder, now)
        if not os.path.exists(archive_base_folder):
            os.makedirs(archive_base_folder)
        if not os.path.exists(archive_folder):
            os.makedirs(archive_folder)
        file = open(os.path.join(self._paths.split_path, '{}.{}'.format(file_name, self._file_extension)), 'w')
        file_archive = open(os.path.join(archive_folder, '{}.{}'.format(file_name, self._file_extension)), 'w')
        file.write(split)
        file_archive.write(split)

    def _read_train_and_val_splits_from_folder(self):
        train = self._read_single_split_from_folder(os.path.join(self._paths.split_path,
                                                                 "split_{}_train.{}".format(self.split_name,
                                                                                            self._file_extension)))
        validation = self._read_single_split_from_folder(
            os.path.join(self._paths.split_path,
                         "split_{}_validation.{}".format(self.split_name, self._file_extension)))
        self.train_paths = train
        self.validation_paths = validation

    @staticmethod
    def _read_single_split_from_folder(file_name):
        if os.path.exists(file_name):
            raise ValueError('{0:} file not found'.format(file_name))
        file = open(file_name, 'r')
        split = file.read().splitlines()
        return split
