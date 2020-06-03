"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020

"""

import numpy as np
import random
from PIL import Image
import copy


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

    return img.astype('uint8')


def tv_clustered_one_hot_to_rgb(one_hot):
    """
    converts the given one hot image to an rgb image given the colors
    :param one_hot: one hot tensor [nx, ny, nr_classes]
    :return: rgb images [nx, ny, 3]
    """

    rgb_img = np.zeros((one_hot.shape[0], one_hot.shape[1], 3))

    idx = np.argmax(one_hot, axis=2)
    n_clusters = one_hot.shape[2]
    for i in range(n_clusters):
        g_val = int(0.0 + i * (255 / n_clusters))
        rgb_img[idx == i] = [g_val, g_val, g_val]

    return rgb_img.astype('uint8')


def one_hot_to_rgb(one_hot, scan):
    """
    converts the given one hot image to an rgb image given the colors
    :param one_hot: one hot tensor [nx, ny, nr_classes]
    :param scan: raw scan used as background for the segmentation
    :return: rgb images [nx, ny, 3]
    """
    # Color for each Label (used for resut Visualization
    seg_label_colors = np.array([
        [0, 0, 0],  # no tumor
        [255, 0, 0],  # necrosis
        [255, 255, 0],  # edmema
        [0, 255, 0],  # non enhancing
        [0, 0, 255]]  #enhancing
    ).astype(float)
    alpha = 0.5

    rgb_img = copy.deepcopy(scan).astype(float)
    # multiclass
    if one_hot.shape[2] > 1:
        idx = np.argmax(one_hot, axis=2)
        for i in range(1, seg_label_colors.shape[0]):
            rgb_img[idx == i] = alpha * rgb_img[idx == i] + (1-alpha) * seg_label_colors[i]
    # binary
    else:
        one_hot = one_hot.reshape((one_hot.shape[0], one_hot.shape[1]))
        rgb_img[one_hot== 1] = alpha * rgb_img[one_hot == 1] + (1-alpha) * seg_label_colors[1]

    return rgb_img.astype('uint8')


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].

    :param data: the array to crop
    :param shape: the target shape
    """
    if data.shape[1] == shape[1] and data.shape[2] == shape[2]:
        return data

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
    :param border: value for the border pixels
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


def combine_img_prediction(data, gt, pred, mode=1):
    """
    Combines the data, grouth thruth and the prediction into one rgb image

    :param data: the data tensor
    :param gt: the ground truth tensor
    :param pred: the prediction tensor
    :param mode: 0 for segmentation 1 for regression
    :returns img: the concatenated rgb image
    """

    ny = data.shape[2]
    ch = data.shape[3]

    data_for_gt = to_rgb(revert_zero_centering(data[:, :, :, 0]).reshape(-1, ny, 1))
    data = np.concatenate((revert_zero_centering(data[:, :, :, 0]),
                           revert_zero_centering(data[:, :, :, 1]),
                           revert_zero_centering(data[:, :, :, 2]),
                           revert_zero_centering(data[:, :, :, 3])),
                          axis=2).reshape(-1, ny*ch, 1)
    data_rgb = to_rgb(data)
    data_size = (data_for_gt.shape[1], data_for_gt.shape[0])
    if mode == 0:
        gt = gt.reshape(-1, gt.shape[2], gt.shape[3])
        pred = pred.reshape(-1, pred.shape[2], pred.shape[3])
        gt_rgb = one_hot_to_rgb(gt, data_for_gt)
        pred_rgb = one_hot_to_rgb(pred, data_for_gt)
    elif mode == 1:
        gts = []
        preds = []
        for i in range(gt.shape[0]):
            index = random.randint(0, gt.shape[-1]-1)
            gts.append(gt[i, :, :, index])
            preds.append(pred[i, :, :, index])
        gt = revert_zero_centering(np.array(gts))
        pred = revert_zero_centering(np.array(preds))
        gt = gt.reshape(-1, gt.shape[2], 1)
        pred = pred.reshape(-1, pred.shape[2], 1)
        gt_rgb = to_rgb(gt)
        pred_rgb = to_rgb(pred)
    else:
        raise ValueError()

    gt_resized = np.array(Image.fromarray(gt_rgb).resize(data_size, Image.NEAREST))
    pred_resized = np.array(Image.fromarray(pred_rgb).resize(data_size, Image.NEAREST))
    img = np.concatenate((data_rgb,
                          gt_resized,
                          pred_resized),
                          axis=1)
    return img


def combine_img_prediction_tvclustering(data, tv, gt, pred):
    """
    Combines the data, grouth thruth and the prediction into one rgb image

    :param data: the data tensor
    :param gt: the ground truth tensor
    :param tv: the tv smoothed tensor
    :param pred: the prediction tensor

    :returns img: the concatenated rgb image
    """

    ny = data.shape[2]
    ch = data.shape[3]
    data = np.concatenate((revert_zero_centering(data[:, :, :, 0]),
                           revert_zero_centering(data[:, :, :, 1]),
                           revert_zero_centering(data[:, :, :, 2]),
                           revert_zero_centering(data[:, :, :, 3])),
                          axis=2).reshape(-1, ny*ch, 1)
    data_rgb = to_rgb(data)

    gts = []
    preds = []
    tvs = []
    nr_cl = int(gt.shape[3] / 4)

    index = random.randint(0, 3)
    tv = tv[:, :, :, index]
    gt = gt[:, :, :, index*nr_cl:(index+1)*nr_cl]
    pred = pred[:, :, :, index*nr_cl:(index+1)*nr_cl]

    gt = np.array(gt).reshape(-1, gt.shape[2], nr_cl)
    pred = np.array(pred).reshape(-1, pred.shape[2], nr_cl)
    tv = revert_zero_centering(np.array(tv)).reshape(-1, tv.shape[2], 1)
    tv_rgb = to_rgb(tv)
    gt_rgb = tv_clustered_one_hot_to_rgb(gt)
    pred_rgb = tv_clustered_one_hot_to_rgb(pred)
    img = np.concatenate((data_rgb,
                          tv_rgb,
                          gt_rgb,
                          pred_rgb),
                          axis=1)
    return img


def revert_zero_centering(data):
    nr_img = data.shape[0]
    images = []
    for i in range(nr_img):
        img = data[i]
        min = np.amin(img)
        img = img - min
        m = np.amax(img)
        if m == 0.0:
            m = 1.0
        img = img / m
        img *= 255.0
        images.append(img)
    return np.array(images)


def get_hard_dice_score(pred, gt, axis=(0, 1, 2, 3), eps=1e-5):
    d = np.sum(gt * pred, axis=axis)
    n1 = np.sum(gt, axis=axis)
    n2 = np.sum(pred, axis=axis)
    dice = (2 * d + eps) / (n1 + n2 + eps)
    dice = np.mean(dice)
    return dice

def get_confusion_metrices(predictions, gt):
    """
    Computes precison, recall and specificity for a given prediction and gt

    :param predictions: input tensor of predictions
    :param gt: tensor of ground truth values

    :returns list of [precision, sensitivity, specificity] op
    """

    tp = float(np.count_nonzero(predictions * gt))
    tn = float(np.count_nonzero((predictions - 1.) * (gt - 1.)))
    fp = float(np.count_nonzero(predictions * (gt - 1.)))
    fn = float(np.count_nonzero((predictions - 1.) * gt))

    precision = (tp + 1e-10) / (tp + fp + 1e-10)
    sensitivity = (tp + 1e-10) / (tp + fn + 1e-10)
    specificity = (tn + 1e-10) / (tn + fp + 1e-10)

    return [precision, sensitivity, specificity]

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return [image_equalized.reshape(image.shape), cdf]

