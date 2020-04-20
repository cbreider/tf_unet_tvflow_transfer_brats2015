"""
Master Thesis Learning Feature Preserving Smoothing as Prior for Image Segmentation
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""

import src.utilities.io_utils as ioutils
import numpy as np
from _thread import start_new_thread, allocate_lock
import multiprocessing
import os
import json
import png
import SimpleITK as sitk


lock = allocate_lock()
thread_finished = None
show_step_size = 4
num_threads = int(multiprocessing.cpu_count())
skip_existing_files = True

ground_truth_path_identifier = ["xx.o.ot", "xx.xx.ot"]
data_dir = "../dataset"
brats_train_dir = os.path.join(data_dir, "BRATS2015_Training")
brats_test_dir = os.path.join(data_dir, "BRATS2015_Testing")
high_grade_gliomas_folder = "HGG"
low_grade_gliomas_folder = "LGG"
test_gg_path = "HGG_LGG"
raw_png_train_dir = os.path.join(data_dir, "2d_slices/png/raw/train")
raw_png_test_dir = os.path.join(data_dir, "2d_slices/png/raw/test")


def get_mha_to_png_file_paths_dict():

    global brats_train_dir
    global brats_test_dir
    global high_grade_gliomas_folder
    global low_grade_gliomas_folder
    global test_gg_path
    global raw_png_test_dir
    global raw_png_train_dir
    mha_to_nrrd_file_dict = dict()
    mha_to_nrrd_file_dict.update(get_paths_dict(brats_train_dir, raw_png_train_dir, gg=high_grade_gliomas_folder))
    mha_to_nrrd_file_dict.update(get_paths_dict(brats_train_dir, raw_png_train_dir, gg=low_grade_gliomas_folder))
    mha_to_nrrd_file_dict.update(get_paths_dict(brats_test_dir, raw_png_test_dir, gg=test_gg_path))
    return mha_to_nrrd_file_dict


def get_paths_dict(base_path_key, base_path_value, ext_key=".mha", ext_val=".png", gg="HGG"):
    global ground_truth_path_identifier
    file_dict = dict()
    gg_path = os.path.join(base_path_key, gg)
    patient_names = os.listdir(gg_path)
    patient_paths = [os.path.join(gg_path, patient) for patient in patient_names]
    file_folders = [[os.path.join(path, tmp_path) for tmp_path in os.listdir(path)] for path in patient_paths]
    flat_list_flat = [item for sublist in file_folders for item in sublist]
    for path in flat_list_flat:
        out_path = path.replace(base_path_key, base_path_value)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for file in os.listdir(path):
            if file.endswith(ext_key):
                file_path_key = os.path.join(path, file)
                file_path_val = file_path_key.replace(base_path_key, base_path_value)
                file_path_val = file_path_val.replace(ext_key, ext_val)
                file_dict[file_path_key] = file_path_val
    return file_dict


def save_3darray_to_pngs(data, filename, skip_if_exists=False, skip_empty=False):
    global ground_truth_path_identifier

    try:
        all = range(np.shape(data)[0])
        data = data.astype(np.uint16)
        non_zero_data = data[np.where(data > 0)]
        values = dict()
        values["mean"] = float(non_zero_data.mean())
        values["max"] = float(non_zero_data.max())
        values["variance"] = float(non_zero_data.var())
        val_path = "{}/values.json".format(filename[:filename.rfind("/")])
        outF = open(val_path, "w")
        outF.write(json.dumps(values))
        outF.close()
        for i in all:
            mha_slice = data[i]
            file_path = "{}_{}.png".format(filename, i)
            if os.path.exists(file_path) and skip_if_exists:
                continue
            if (not(float(mha_slice.max()) == 0.0 and skip_empty)) or (ground_truth_path_identifier[0] in filename.lower() or
                                                 ground_truth_path_identifier[1] in filename.lower()):
                with open(file_path, 'wb') as f:
                    w = png.Writer(240, 240, greyscale=True, bitdepth=16)
                    w.write(f, mha_slice)
                    f.close()
    except Exception as e:
        print("type error: " + str(e))


def load_3d_volume_as_array(filename):
    if ('.mha' in filename):
        return load_mha_volume_as_array(filename)
    raise ValueError('{0:} unspported file format'.format(filename))


def load_mha_volume_as_array(filename):
    img = sitk.ReadImage(filename)
    nda = sitk.GetArrayFromImage(img)
    return nda


def run_mha_to_sclice_conversion(file_dict_part, thread_num):
    global show_step_size, threads_running, skip_empty_files
    length = len(file_dict_part)
    tmp = 1
    for key, value in file_dict_part.items():
        arr = load_mha_volume_as_array(key)
        save_3darray_to_pngs(arr, value, skip_if_exists=True)
        if tmp % show_step_size == 0:
            print("Saved scan {} of {} as nrrd in tread {}".format(tmp, length, thread_num))
        tmp += 1
    threads_running[thread_num - 1] = False


if __name__ == "__main__":
    mha_file_dict = get_mha_to_png_file_paths_dict()
    complete_dict_len = len(mha_file_dict)
    chunked_dict = ioutils.chunk_dict(mha_file_dict, num_threads)
    threads_running = [True for el in range(num_threads)]
    print("Starting conversion from mha to nrrd in {} threads on {} elements...".format(num_threads, complete_dict_len))
    for i in range(num_threads):
        print("Starting thread {} with {} elements".format(i + 1, len(chunked_dict[i])))
        start_new_thread(run_mha_to_sclice_conversion, (chunked_dict[i], i + 1))

    while any(thread_running for thread_running in threads_running):
        pass
    print("Finished conversion from mha to nrrd!")