"""
Author: Christian Breiderhoff
"""

import os

is_loaded = False
arr_ext = ".nrrd"
high_grade_gliomas_folder = 'HGG'
low_grade_gliomas_folder = 'LGG'
test_gg_path = "{}_{}".format(high_grade_gliomas_folder, low_grade_gliomas_folder)
ground_truth_path_identifier = ".xx.o.ot."
t1_identifier ="mr_t1"
t1c_identifier ="mr_t1c"
t2_identifier ="mr_t2"
flair_identifier ="mr_flair"
project_dir = None
data_dir = "../dataset"
slice_dir = "2d_slices"
raw_train_dir = "train"
raw_test_dir = "test"
tv_flow_out_dir = "tvflow"
raw_dir = "raw"
brats_train_dir = "BRATS2015_Training"
brats_test_dir = "BRATS2015_Testing"
split_path = "splits"
checkpoint_path = "tf_checkpoints"
summary_path = "tf_summaries"


def load_data_paths():
    global project_dir, data_dir,  slice_dir, raw_train_dir, raw_test_dir, tv_flow_out_dir, raw_dir, brats_train_dir, \
        brats_test_dir, is_loaded, raw_dir, split_path, checkpoint_path, summary_path

    project_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoint_path = os.path.join(project_dir, checkpoint_path)
    summary_path = os.path.join(project_dir, summary_path)
    data_dir = os.path.join(project_dir, data_dir)
    split_path = os.path.join(data_dir, split_path)
    brats_train_dir = os.path.join(data_dir, brats_train_dir)
    brats_test_dir = os.path.join(data_dir, brats_test_dir)
    slice_dir = os.path.join(data_dir, slice_dir)
    raw_dir = os.path.join(slice_dir, raw_dir)
    tv_flow_out_dir = os.path.join(slice_dir, tv_flow_out_dir)
    raw_train_dir = os.path.join(raw_dir, raw_train_dir)
    raw_test_dir = os.path.join(raw_dir, raw_test_dir)

    if not os.path.exists(data_dir):
        return
    if not os.path.exists(brats_train_dir):
        return
    if not os.path.exists(brats_test_dir):
        return
    if not os.path.exists(slice_dir):
        os.makedirs(slice_dir)
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    if not os.path.exists(tv_flow_out_dir):
        os.makedirs(tv_flow_out_dir)
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(tv_flow_out_dir):
        os.makedirs(tv_flow_out_dir)
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(raw_test_dir):
        os.makedirs(raw_test_dir)
    if not os.path.exists(raw_train_dir):
        os.makedirs(raw_train_dir)

    is_loaded = True
