"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
"""

import os


class DataPaths(object):

    def __init__(self, data_path = "default"):
        self.is_loaded = False
        self.arr_ext = ".nrrd"
        self.high_grade_gliomas_folder = 'HGG'
        self.low_grade_gliomas_folder = 'LGG'
        self.test_gg_path = "{}_{}".format(self.high_grade_gliomas_folder, self.low_grade_gliomas_folder)
        self.ground_truth_path_identifier = ".xx.o.ot."
        self.t1_identifier = "mr_t1"
        self.t1c_identifier = "mr_t1c"
        self.t2_identifier = "mr_t2"
        self.flair_identifier = "mr_flair"
        self.project_dir = None
        self.data_dir = data_path
        self.slice_dir = "2d_slices"
        self.raw_train_dir = "train"
        self.raw_test_dir = "test"
        self.tv_flow_out_dir = "tvflow"
        self.raw_dir = "raw"
        self.brats_train_dir = "BRATS2015_Training"
        self.brats_test_dir = "BRATS2015_Testing"
        self.split_path = "splits"
        self.checkpoint_path = "tf_checkpoints"
        self.summary_path = "tf_summaries"

    def _load_data_paths(self):
        self.project_dir = os.path.dirname(os.path.realpath(__file__))
        self.checkpoint_path = os.path.join(self.project_dir, self.checkpoint_path)
        self.summary_path = os.path.join(self.project_dir, self.summary_path)
        if not self.data_dir == "default":
            self.data_dir = os.path.join(self.data_dir, self.data_dir)
        else:
            self.data_dir = os.path.join("../dataset", self.data_dir)
        self.split_path = os.path.join(self.data_dir, self.split_path)
        self.brats_train_dir = os.path.join(self.data_dir, self.brats_train_dir)
        self.brats_test_dir = os.path.join(self.data_dir, self.brats_test_dir)
        self.slice_dir = os.path.join(self.data_dir, self.slice_dir)
        self.raw_dir = os.path.join(self.slice_dir, self.raw_dir)
        self.tv_flow_out_dir = os.path.join(self.slice_dir, self.tv_flow_out_dir)
        self.raw_train_dir = os.path.join(self.raw_dir, self.raw_train_dir)
        self.raw_test_dir = os.path.join(self.raw_dir, self.raw_test_dir)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError()
        if not os.path.exists(self.brats_train_dir):
            return FileNotFoundError()
        if not os.path.exists(self.brats_test_dir):
            raise FileNotFoundError()
        if not os.path.exists(self.slice_dir):
            os.makedirs(self.slice_dir)
        if not os.path.exists(self.split_path):
            os.makedirs(self.split_path)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        if not os.path.exists(self.tv_flow_out_dir):
            os.makedirs(self.tv_flow_out_dir)
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        if not os.path.exists(self.tv_flow_out_dir):
            os.makedirs(self.tv_flow_out_dir)
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        if not os.path.exists(self.raw_test_dir):
            os.makedirs(self.raw_test_dir)
        if not os.path.exists(self.raw_train_dir):
            os.makedirs(self.raw_train_dir)
        self.is_loaded = True
