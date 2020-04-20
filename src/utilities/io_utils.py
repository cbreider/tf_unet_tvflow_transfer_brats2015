"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""

import sys
import os
import numpy as np
from PIL import Image
import SimpleITK as sitk


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer in the terminal.

    :param question: is a string that is presented to the user.
    :param default:  is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    :returns The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def save_image(img, path):
    """
    Writes the image to disk

    :param img: the rgb image to save
    :param path: the target path
    """
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300,300], quality=90)


def create_training_path(output_path, prefix="run_"):
    """
    Enumerates a new path using the prefix under the given output_path
    :param output_path: the root path
    :param prefix: (optional) defaults to `run_`
    :return: the generated path as string in form `output_path`/`prefix_` + `<number>`
    """
    idx = 0
    path = os.path.join(output_path, "{:}{:03d}".format(prefix, idx))
    while os.path.exists(path):
        idx += 1
        path = os.path.join(output_path, "{:}{:03d}".format(prefix, idx))
    return path


# supported file extensions
nrrd_ext = ".nrrd"
mha_ext = ".mha"
npy_ext = ".npy"


def chunk_dict(dict_in, num_seq):
    avg = len(dict_in) / float(num_seq)
    out = []
    for seq in range(num_seq):
        out.append(dict())
    last = avg
    i = 0
    seq_counter = 0
    for k, v in dict_in.items():
        out[seq_counter][k] = v
        if i >= int(last):
            seq_counter += 1
            last += avg
        i += 1

    return out


def chunk_list_size(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunk_list(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def shuffle_lists(list_one, list_two):
    """Conjoined shuffling of the list of paths and labels."""
    if len(list_one) == len(list_two):
        raise ValueError("Lists must have the same length {} != {}".format(len(list_one), len(list_two)))

    permutation = np.random.permutation(list_two)
    list_one_s = []
    list_two_s = []
    for i in permutation:
        list_one_s.append(list_one[i])
        list_two_s.append(list_two[i])

    return list_one, list_two


def load_dataset_from_mha_files(file_paths, skip_empty=True):
    """
    loads mha volumes input and gt given by the file paths dict and converts them to single sclices
    :param file_paths:
    :param skip_empty:
    :return:
    """
    out = []
    f=0
    for in_f, gt_f in file_paths.items():
        gt = load_3d_volume_as_array(gt_f)
        in_vol = load_3d_volume_as_array(in_f)
        gt = gt.astype(np.uint8)
        in_vol = in_vol.astype(np.float32) / np.float32(np.max(in_vol)) * 255.0
        in_vol = in_vol.astype(np.uint8)
        for i in range(gt.shape[0]):
            if int(np.max(in_vol[i]) == 0) and skip_empty:
                continue
            out.append([in_vol[i], gt[i]])
        f+=1
    return out


def search_file_in_folder_list(folder_list, file_name):
    """
    Find the full filename from a list of folders
    inputs:
        folder_list: a list of folders
        file_name:  filename
    outputs:
        full_file_name: the full filename
    """
    file_exist = False
    for folder in folder_list:
        full_file_name = os.path.join(folder, file_name)
        if os.path.isfile(full_file_name):
            file_exist = True
            break
    if not file_exist:
        raise ValueError('{0:} is not found in {1:}'.format(file_name, folder))
    return full_file_name


def load_2d_volume_as_array(filename):
    if npy_ext in filename:
        data = load_npy_file_as_array(filename=filename)
    #elif nrrd_ext in filename:
    #    data = load_nrrd_file_as_array(filename=filename, include_header=False)
    else:
        raise ValueError('{0:} unsupported file format'.format(filename))
    if len(data.shape) != 2:
        raise ValueError("{0:} Loaded array is not 2dim")
    return data


def load_3d_volume_as_array(filename):
    if mha_ext in filename:
        data = load_mha_volume_as_array(filename)
    #elif nrrd_ext in filename:
        #data = load_nrrd_file_as_array(filename=filename, include_header=False)
    elif npy_ext in filename:
        data = load_npy_file_as_array(filename=filename)
    else:
        raise ValueError('{0:} unsupported file format'.format(filename))
    if len(data.shape) != 3:
        raise ValueError("{0:} Loaded array is not 3dim")
    data = data.astype(dtype=np.int16)
    return data


"""
def load_nrrd_file_as_array(filename, include_header=False):
    if not (nrrd_ext in filename):
        raise ValueError('{0:} unsupported file format'.format(filename))
    readdata, header = nrrd.read(filename)
    if include_header:
        return np.array(readdata), header
    else:
        return np.array(readdata)
"""


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


def load_npy_file_as_array(filename):
    if not (npy_ext in filename):
        raise ValueError('{0:} unsupported file format'.format(filename))
    return np.load(filename)


def load_mha_volume_as_array(filename):
    if not (mha_ext in filename):
        raise ValueError('{0:} unsupported file format'.format(filename))
    img = sitk.ReadImage(filename)
    nda = np.array(sitk.GetArrayFromImage(img))
    return nda


def save_scan_as_mha(scan, filepath, type=np.int16):
    data = scan.astype(type)
    img = sitk.GetImageFromArray(data)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(filepath)
    writer.Execute(img)


def save_array_as_nifty_volume(data, filename, reference_name=None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if reference_name is not None:
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)
