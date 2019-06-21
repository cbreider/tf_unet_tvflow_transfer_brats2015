"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
"""


import os
import numpy as np
import random
from scipy import ndimage
import SimpleITK as sitk
import nrrd
from PIL import Image


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
    elif nrrd_ext in filename:
        data = load_nrrd_file_as_array(filename=filename, include_header=False)
    else:
        raise ValueError('{0:} unsupported file format'.format(filename))
    if len(data.shape) != 2:
        raise ValueError("{0:} Loaded array is not 2dim")
    return data


def load_3d_volume_as_array(filename):
    if mha_ext in filename:
        data = load_mha_volume_as_array(filename)
    elif nrrd_ext in filename:
        data = load_nrrd_file_as_array(filename=filename, include_header=False)
    elif npy_ext in filename:
        data = load_npy_file_as_array(filename=filename)
    else:
        raise ValueError('{0:} unsupported file format'.format(filename))
    if len(data.shape) != 3:
        raise ValueError("{0:} Loaded array is not 3dim")
    return data


def load_nrrd_file_as_array(filename, include_header=False):
    if not (nrrd_ext in filename):
        raise ValueError('{0:} unsupported file format'.format(filename))
    readdata, header = nrrd.read(filename)
    if include_header:
        return readdata, header
    else:
        return readdata


def load_npy_file_as_array(filename):
    if not (npy_ext in filename):
        raise ValueError('{0:} unsupported file format'.format(filename))
    return np.load(filename)


def load_mha_volume_as_array(filename):
    if not (mha_ext in filename):
        raise ValueError('{0:} unsupported file format'.format(filename))
    img = sitk.ReadImage(filename)
    nda = sitk.GetArrayFromImage(img)
    return nda


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


def intensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out


def get_nd_bounding_box(label, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if type(margin) is int:
        margin = [margin] * len(input_shape)
    assert (len(input_shape) == len(margin))
    indexes = np.nonzero(label)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indexes[i].min())
        idx_max.append(indexes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max


def crop_nd_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert (2 <= dim <= 5)
    if dim == 2:
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif dim == 3:
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif dim == 4:
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif dim == 5:
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output


def set_nd_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume):
    """
    set a subregion to an nd image.
    """
    dim = len(bb_min)
    out = volume
    if dim == 2:
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1))] = sub_volume
    elif dim == 3:
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1))] = sub_volume
    elif dim == 4:
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1),
                   range(bb_min[3], bb_max[3] + 1))] = sub_volume
    else:
        raise ValueError("array dimension should be 2, 3 or 4")
    return out


def convert_label(in_volume, label_convert_source, label_convert_target):
    """
    convert the label value in a volume
    inputs:
        in_volume: input nd volume with label set label_convert_source
        label_convert_source: a list of integers denoting input labels, e.g., [0, 1, 2, 4]
        label_convert_target: a list of integers denoting output labels, e.g.,[0, 1, 2, 3]
    outputs:
        out_volume: the output nd volume with label set label_convert_target
    """
    mask_volume = np.zeros_like(in_volume)
    convert_volume = np.zeros_like(in_volume)
    for i in range(len(label_convert_source)):
        source_lab = label_convert_source[i]
        target_lab = label_convert_target[i]
        if source_lab != target_lab:
            temp_source = np.asarray(in_volume == source_lab)
            temp_target = target_lab * temp_source
            mask_volume = mask_volume + temp_source
            convert_volume = convert_volume + temp_target
    out_volume = in_volume * 1
    out_volume[mask_volume > 0] = convert_volume[mask_volume > 0]
    return out_volume


def get_random_roi_sampling_center(input_shape, output_shape, sample_mode, bounding_box=None):
    """
    get a random coordinate representing the center of a roi for sampling
    inputs:
        input_shape: the shape of sampled volume
        output_shape: the desired roi shape
        sample_mode: 'valid': the entire roi should be inside the input volume
                     'full': only the roi centre should be inside the input volume
        bounding_box: the bounding box which the roi center should be limited to
    outputs:
        center: the output center coordinate of a roi
    """
    center = []
    for i in range(len(input_shape)):
        if sample_mode[i] == 'full':
            if bounding_box:
                x0 = bounding_box[i * 2]
                x1 = bounding_box[i * 2 + 1]
            else:
                x0 = 0
                x1 = input_shape[i]
        else:
            if bounding_box:
                x0 = bounding_box[i * 2] + int(output_shape[i] / 2)
                x1 = bounding_box[i * 2 + 1] - int(output_shape[i] / 2)
            else:
                x0 = int(output_shape[i] / 2)
                x1 = input_shape[i] - x0
        if x1 <= x0:
            centeri = int((x0 + x1) / 2)
        else:
            centeri = random.randint(x0, x1)
        center.append(centeri)
    return center


def transpose_volumes(volumes, slice_direction):
    """
    transpose a list of volumes
    inputs:
        volumes: a list of nd volumes
        slice_direction: 'axial', 'sagittal', or 'coronal'
    outputs:
        tr_volumes: a list of transposed volumes
    """
    if slice_direction == 'axial':
        tr_volumes = volumes
    elif slice_direction == 'sagittal':
        tr_volumes = [np.transpose(x, (2, 0, 1)) for x in volumes]
    elif slice_direction == 'coronal':
        tr_volumes = [np.transpose(x, (1, 0, 2)) for x in volumes]
    else:
        print('undefined slice direction:', slice_direction)
        tr_volumes = volumes
    return tr_volumes


def resize_nd_volume_to_given_shape(volume, out_shape, order=3):
    """
    resize an nd volume to a given shape
    inputs:
        volume: the input nd volume, an nd array
        out_shape: the desired output shape, a list
        order: the order of interpolation
    outputs:
        out_volume: the reized nd volume with given shape
    """
    shape0 = volume.shape
    assert (len(shape0) == len(out_shape))
    scale = [(out_shape[i] + 0.0) / shape0[i] for i in range(len(shape0))]
    out_volume = ndimage.interpolation.zoom(volume, scale, order=order)
    return out_volume


def extract_roi_from_volume(volume, in_center, output_shape, fill='random'):
    """
    extract a roi from a 3d volume
    inputs:
        volume: the input 3D volume
        in_center: the center of the roi
        output_shape: the size of the roi
        fill: 'random' or 'zero', the mode to fill roi region where is outside of the input volume
    outputs:
        output: the roi volume
    """
    input_shape = volume.shape
    if fill == 'random':
        output = np.random.normal(0, 1, size=output_shape)
    else:
        output = np.zeros(output_shape)
    r0max = [int(x / 2) for x in output_shape]
    r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
    out_center = r0max

    output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                  range(out_center[1] - r0[1], out_center[1] + r1[1]),
                  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
                      range(in_center[1] - r0[1], in_center[1] + r1[1]),
                      range(in_center[2] - r0[2], in_center[2] + r1[2]))]
    return output


def set_roi_to_volume(volume, center, sub_volume):
    """
    set the content of an roi of a 3d/4d volume to a sub volume
    inputs:
        volume: the input 3D/4D volume
        center: the center of the roi
        sub_volume: the content of sub volume
    outputs:
        output_volume: the output 3D/4D volume
    """
    volume_shape = volume.shape
    patch_shape = sub_volume.shape
    output_volume = volume
    for i in range(len(center)):
        if center[i] >= volume_shape[i]:
            return output_volume
    r0max = [int(x / 2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - center[i]) for i in range(len(r0max))]
    patch_center = r0max

    if len(center) == 3:
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif len(center) == 4:
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]),
                             range(center[3] - r0[3], center[3] + r1[3]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                              range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")
    return output_volume


def get_largest_two_component(img, print_info=False, threshold=None):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 3D volume
        threshold: a size threshold
    outputs:
        out_img: the output volume
    """
    s = ndimage.generate_binary_structure(3, 2)  # iterate structure
    labeled_array, numpatches = ndimage.label(img, s)  # labeling
    sizes = ndimage.sum(img, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if print_info:
        print('component size', sizes_list)
    if len(sizes) == 1:
        out_img = img
    else:
        if threshold:
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if temp_size > threshold:
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if max_size2 * 10 > max_size1:
                component1 = (component1 + component2) > 0
            out_img = component1
    return out_img


def fill_holes(img):
    """
    filling small holes of a binary volume with morphological operations
    """
    neg = 1 - img
    s = ndimage.generate_binary_structure(3, 1)  # iterate structure
    labeled_array, numpatches = ndimage.label(neg, s)  # labeling
    sizes = ndimage.sum(neg, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    max_size = sizes_list[-1]
    max_label = np.where(sizes == max_size)[0] + 1
    component = labeled_array == max_label
    return 1 - component


def remove_external_core(lab_main, lab_ext):
    """
    remove the core region that is outside of whole tumor
    """
    # for each component of lab_ext, compute the overlap with lab_main
    s = ndimage.generate_binary_structure(3, 2)  # iterate structure
    labeled_array, num_patches = ndimage.label(lab_ext, s)  # labeling
    sizes = ndimage.sum(lab_ext, labeled_array, range(1, num_patches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    new_lab_ext = np.zeros_like(lab_ext)
    for i in range(len(sizes)):
        size_i = sizes_list[i]
        label_i = np.where(sizes == size_i)[0] + 1
        component_i = labeled_array == label_i
        overlap = component_i * lab_main
        if (overlap.sum() + 0.0) / size_i >= 0.5:
            new_lab_ext = np.maximum(new_lab_ext, component_i)
    return new_lab_ext


def binary_dice3d(s, g):
    """
    dice score of 3d binary volumes
    inputs:
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert (len(s.shape) == 3)
    assert (s.shape == g.shape)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0 * s0 + 1e-10) / (s1 + s2 + 1e-10)

    return dice


def plot_prediction(x_test, y_test, prediction, save=False):
    import matplotlib
    import matplotlib.pyplot as plt

    test_size = x_test.shape[0]
    fig, ax = plt.subplots(test_size, 3, figsize=(12,12), sharey=True, sharex=True)

    x_test = crop_to_shape(x_test, prediction.shape)
    y_test = crop_to_shape(y_test, prediction.shape)

    ax = np.atleast_2d(ax)
    for i in range(test_size):
        cax = ax[i, 0].imshow(x_test[i])
        plt.colorbar(cax, ax=ax[i,0])
        cax = ax[i, 1].imshow(y_test[i, ..., 1])
        plt.colorbar(cax, ax=ax[i,1])
        pred = prediction[i, ..., 1]
        pred -= np.amin(pred)
        pred /= np.amax(pred)
        cax = ax[i, 2].imshow(pred)
        plt.colorbar(cax, ax=ax[i,2])
        if i==0:
            ax[i, 0].set_title("x")
            ax[i, 1].set_title("y")
            ax[i, 2].set_title("pred")
    fig.tight_layout()

    if save:
        fig.savefig(save)
    else:
        fig.show()
        plt.show()


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [nx, ny, channels]

    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    if np.amax(img) != 0:
        img /= np.amax(img)

    img *= 255
    return img


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].

    :param data: the array to crop
    :param shape: the target shape
    """
    diff_nx = (data.shape[1] - shape[1])
    diff_ny = (data.shape[2] - shape[2])

    offset_nx_left = diff_nx // 2
    offset_nx_right = diff_nx - offset_nx_left
    offset_ny_left = diff_ny // 2
    offset_ny_right = diff_ny - offset_ny_left

    cropped = data[:, offset_nx_left:(-offset_nx_right), offset_ny_left:(-offset_ny_right)]

    assert cropped.shape[1] == shape[1]
    assert cropped.shape[2] == shape[2]
    return cropped


def expand_to_shape(data, shape, border=0):
    """
    Expands the array to the given image shape by padding it with a border (expects a tensor of shape [batches, nx, ny, channels].

    :param data: the array to expand
    :param shape: the target shape
    """
    diff_nx = shape[1] - data.shape[1]
    diff_ny = shape[2] - data.shape[2]

    offset_nx_left = diff_nx // 2
    offset_nx_right = diff_nx - offset_nx_left
    offset_ny_left = diff_ny // 2
    offset_ny_right = diff_ny - offset_ny_left

    expanded = np.full(shape, border, dtype=np.float32)
    expanded[:, offset_nx_left:(-offset_nx_right), offset_ny_left:(-offset_ny_right)] = data

    return expanded


def combine_img_prediction(data, gt, pred):
    """
    Combines the data, grouth thruth and the prediction into one rgb image

    :param data: the data tensor
    :param gt: the ground thruth tensor
    :param pred: the prediction tensor

    :returns img: the concatenated rgb image
    """
    ny = pred.shape[2]
    ch = data.shape[3]
    img = np.concatenate((to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)),
                          to_rgb(crop_to_shape(gt[..., 1], pred.shape).reshape(-1, ny, 1)),
                          to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
    return img


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