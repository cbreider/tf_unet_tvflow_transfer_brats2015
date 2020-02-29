"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""

import os
from datetime import datetime
from shutil import copyfile
import logging


class DataPaths(object):

    def __init__(self, data_path="default", mode="TVFLOW_REGRESSION", tumor_mode="COMPLETE"):
        self.is_loaded = False
        self.nrrd_ext = ".nrrd"
        self.png_ext = ".png"
        self.mha_ext = ".mha"
        self.high_grade_gliomas_folder = 'HGG'
        self.low_grade_gliomas_folder = 'LGG'
        self.test_gg_path = "{}_{}".format(self.high_grade_gliomas_folder, self.low_grade_gliomas_folder)
        self.ground_truth_path_identifier = ["xx.o.ot", "xx.xx.ot"]
        self.t1_identifier = "mr_t1"
        self.t1c_identifier = "mr_t1c"
        self.t2_identifier = "mr_t2"
        self.flair_identifier = "mr_flair"
        self.project_dir = None
        self.data_dir = data_path
        self.slice_dir = "2d_slices"
        self.png_dir = "png"
        self.nrrd_dir = "nrrd"
        self.raw_train_dir = "train"
        self.raw_test_dir = "test"
        self.tv_flow_out_dir = "tvflow"
        self.raw_dir = "raw"
        self.brats_train_dir = "BRATS2015_Training"
        self.brats_test_dir = "BRATS2015_Testing/HGG_LGG"
        self.split_path = "splits"
        self.tf_out_path = "tf_model_output"
        self.mode = mode
        self.tumor_mode = tumor_mode

    def load_data_paths(self, mkdirs=True, restore_dir=None):
        self.project_dir = os.getcwd()  # os.path.dirname(os.path.realpath(__file__))
        self.tf_out_path = os.path.join(self.project_dir, self.tf_out_path)
        if not self.data_dir == "default":
            self.data_dir = self.data_dir
        else:
            self.data_dir = os.path.join(self.project_dir, "../dataset")
        self.split_path = os.path.join(self.data_dir, self.split_path)
        self.brats_train_dir = os.path.join(self.data_dir, self.brats_train_dir)
        self.brats_test_dir = os.path.join(self.data_dir, self.brats_test_dir)
        self.slice_dir = os.path.join(self.data_dir, self.slice_dir)
        self.png_dir = os.path.join(self.slice_dir, self.png_dir)
        self.raw_dir = os.path.join(self.png_dir, self.raw_dir)
        self.tv_flow_out_dir = os.path.join(self.png_dir, self.tv_flow_out_dir)
        self.raw_train_dir = os.path.join(self.raw_dir, self.raw_train_dir)
        self.raw_test_dir = os.path.join(self.raw_dir, self.raw_test_dir)

        self.tf_out_path = os.path.join(self.project_dir, self.tf_out_path)
        tf_out_path_tmp = "{0:%Y-%m-%d_%H:%M:%S}".format(datetime.now())
        tf_out_path_tmp = "{}_{}_{}".format(self.mode, self.tumor_mode, tf_out_path_tmp)

        self.tf_out_path = os.path.join(self.tf_out_path, tf_out_path_tmp)

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError()
        #if not os.path.exists(self.brats_train_dir):
        #    return FileNotFoundError()
        #if not os.path.exists(self.brats_test_dir):
        #    raise FileNotFoundError()
        if not os.path.exists(self.slice_dir):
            raise FileNotFoundError()
        if not os.path.exists(self.png_dir):
            raise FileNotFoundError()
        if restore_dir is None and mkdirs:
            if not os.path.exists(self.tf_out_path):
                os.makedirs(self.tf_out_path)
                # save config file to out folder
        else:
            self.tf_out_path = restore_dir

        copyfile("configuration.py", os.path.join(self.tf_out_path,
                                                  "configuration.py.{0:%Y-%m-%d_%H:%M:%S}".format(datetime.now())))
        self.is_loaded = True
