"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019
"""


import tensorflow as tf
import numpy as np


def preprocess_images(scan, ground_truth):
    """
    combined pre processing input and gt images


    :param scan: input image
    :param ground_truth: ground_truth image
    :returns:cobmbined padded,cropped and flipped images
    """
    #  preprocess the image
    combined = tf.concat([scan, ground_truth], axis=2)
    image_shape = tf.shape(scan)

    last_label_dim = tf.shape(ground_truth)[-1]
    last_image_dim = tf.shape(scan)[-1]
    size = tf.random.uniform((),
                             minval=tf.cast(tf.math.divide(tf.cast(image_shape[0], tf.float32),
                                                         tf.constant(2.0)), tf.int32),
                             maxval=image_shape[0],
                             dtype=tf.int32)
    combined_crop = tf.random_crop(value=combined,
                                   size=tf.concat([[size, size], [last_label_dim + last_image_dim]], axis=0))
    combined_flip = tf.image.random_flip_left_right(combined_crop)
    im = tf.image.resize_images(combined_flip[:, :, :last_image_dim], size=[image_shape[0], image_shape[1]])
    gt = tf.image.resize_images(combined_flip[:, :, last_image_dim:], size=[image_shape[0], image_shape[1]])
    return im, gt


def crop_images_to_to_non_zero(scan, ground_truth, size, tvimg=None):
    """
    crops input and gt images to bounding box of non zero area of gt image


    :param scan: input image
    :param ground_truth: ground_truth image
    :returns:cobmbined padded,cropped and flipped images
    """
    # HACK check if gt is completly zero then return orginal images
    total = tf.reduce_sum(tf.abs(ground_truth))
    is_all_zero = tf.equal(total, 0)
    return tf.cond(is_all_zero,
                   lambda: (scan, ground_truth, tvimg),
                   lambda: crop_non_zero_internal(scan=scan, ground_truth=ground_truth, tvimg=tvimg, out_size=size))


def crop_non_zero_internal(scan, ground_truth, out_size, tvimg=None):
    resize_tv = None
    crop_tv = None
    scan = tf.cast(scan, tf.int32)
    zero = tf.constant(0, dtype=tf.int32)
    where = tf.not_equal(scan, zero)
    indices = tf.where(where)
    min_y = tf.reduce_min(indices[:, 0])
    min_x = tf.reduce_min(indices[:, 1])
    max_y = tf.reduce_max(indices[:, 0])
    max_x = tf.reduce_max(indices[:, 1])
    height = tf.math.add(tf.math.subtract(max_y, min_y), tf.convert_to_tensor(1, dtype=tf.int64))
    width = tf.math.add(tf.math.subtract(max_x, min_x), tf.convert_to_tensor(1, dtype=tf.int64))
    crop_in = tf.image.crop_to_bounding_box(scan, min_y, min_x, height, width)
    crop_gt = tf.image.crop_to_bounding_box(ground_truth, min_y, min_x, height, width)
    if tvimg is not None:
        crop_tv = tf.image.crop_to_bounding_box(tvimg, min_y, min_x, height, width)
    resize_in = tf.image.resize_images(crop_in, out_size,
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    resize_gt = tf.image.resize_images(crop_gt, out_size,
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if crop_tv is not None:
        resize_tv = tf.cast(tf.image.resize_images(crop_tv, out_size,
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), tf.float32)
    return tf.cast(resize_in, tf.float32), tf.cast(resize_gt, tf.float32), resize_tv


def load_png_image(filename, nr_channels, img_size, data_type=tf.float32):
    """
    Loads a png images within a TF pipeline

    :param filename: use scale images from tvflow as gt instead of smoothed images
    :param data_type: data type in which the image is casted
    :param nr_channels: number of channels in the image
    :param img_size: size of the image
    :returns: image of size=size, nr of channels=nr_channels, dtype = data_type
    """
    try:
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=nr_channels)
        img_resized = tf.image.resize_images(img_decoded, size=img_size)
        img = tf.cast(img_resized, data_type)
        return img
    except Exception as e:
        print("type error: " + str(e) + str(filename))


def get_tv_smoothed_and_meanshift_clusterd_one_hot(image, tv_tau, tv_weight, tv_eps, tv_m_itr, ms_itr=-1, win_r=0.05,
                                                   n_clusters=10):

    tv_sm = get_tv_smoothed(image, tv_tau, tv_weight, tv_eps, tv_m_itr)
    tv_sm = tv_sm - tf.reduce_min(tv_sm)
    mean, var = tf.nn.moments(tv_sm, axes=[0, 1, 2])
    tv_sm = tf.math.divide((tv_sm), tf.math.sqrt(var))
    tv_sm = tv_sm / tf.reduce_max(tv_sm)

    c, __ = mean_shift(input_x=tf.reshape(tv_sm, (-1, 1)), bin_seed=True,
                       n_updates=ms_itr, window_radius=win_r)
    c = reshape_clusters_to_cluster_number(c, n_clusters)
    centroids_expanded = tf.expand_dims(c, 1)
    centroids_expanded = tf.transpose(centroids_expanded)
    centroids_expanded = tf.expand_dims(centroids_expanded, 0)
    distances = tf.square(tf.subtract(image, centroids_expanded))
    assignments = tf.cast(tf.expand_dims(tf.argmin(distances, axis=2, output_type=tf.int32), 2), tf.float32)
    return tv_sm, assignments


def get_tv_smoothed_and_kmeans_clusterd_one_hot(image, nr_img, tv_tau, tv_weight, tv_eps, tv_m_itr, km_cluster_n, km_itr_n):

    tv_sm = get_tv_smoothed(image, tv_tau, tv_weight, tv_eps, tv_m_itr)
    tv_sm = tv_sm - tf.reduce_min(tv_sm)
    mean, var = tf.nn.moments(tv_sm, axes=[0, 1, 2])
    tv_sm = tf.math.divide((tv_sm), tf.math.sqrt(var))
    tv_sm = tv_sm / tf.reduce_max(tv_sm)

    clustered = get_kmeans(tv_sm, clusters_n=km_cluster_n, iteration_n=km_itr_n)
    #tv_sm = tf.expand_dims(tv_sm, 2)
    clustered = tf.expand_dims(clustered, 2)
    return tv_sm, clustered


def get_tv_smoothed(img, tau, weight, eps, m_itr):
    inimg = img - tf.reduce_min(img)
    mean, var = tf.nn.moments(inimg, axes=[0, 1, 2])
    inimg = tf.math.divide((inimg), tf.math.sqrt(var))
    inimg = inimg / tf.reduce_max(inimg)

    u = tf.zeros_like(inimg)
    px = tf.zeros_like(inimg)
    py = tf.zeros_like(inimg)
    nm = tf.cast(tf.shape(img)[0] * tf.shape(img)[1] * tf.shape(img)[2], tf.float32)
    error = tf.constant(0.0)
    err_prev = tf.constant(eps)
    err_init = tf.constant(0.0)
    i = tf.constant(0)

    def _tv_cond(img, u, px, py, tau, weight, nm, error, err_prev, err_init, eps, i, m_itr):
        return tf.logical_or(tf.greater_equal(tf.math.abs(err_prev - error), tf.multiply(eps, err_init)),
                             tf.less_equal(i, m_itr))

    def _tv_body(img, u, px, py, tau, weight, nm, error, err_prev, err_init, eps, i, m_itr):
        u_old = u
        ux = tf.subtract(tf.roll(u, shift=-1, axis=1), u)
        uy = tf.subtract(tf.roll(u, shift=-1, axis=0), u)
        px_new = tf.add(px, tf.multiply(tf.truediv(tau, weight), ux))
        py_new = tf.add(py, tf.multiply(tf.truediv(tau, weight), uy))
        norm_new = tf.math.maximum(tf.constant(1.0), tf.math.sqrt(
            tf.math.pow(px_new, tf.constant(2.0)) + tf.math.pow(py_new, tf.constant(2.0))))
        px = tf.truediv(px_new, norm_new)
        py = tf.truediv(py_new, norm_new)
        # calculate divergence
        rx = tf.roll(px, shift=1, axis=1)
        ry = tf.roll(py, shift=1, axis=0)
        div_p = tf.add(tf.subtract(px, rx), tf.subtract(py, ry))

        # update image
        u = tf.add(img, tf.multiply(weight, div_p))
        err_prev = error
        # calculate error
        error = tf.truediv(tf.norm(tf.subtract(u, u_old)), tf.math.sqrt(nm))
        err_init = tf.cond(tf.equal(i, tf.constant(0)), lambda: error, lambda: err_init)
        i = tf.math.add(i, tf.constant(1))
        return [img, u, px, py, tau, weight, nm, error, err_prev, err_init, eps, i, m_itr]

    inimg, u, px, py, tau, weight, nm, error, err_prev, err_init, eps, i, m_itr = tf.while_loop(
        _tv_cond,
        _tv_body,
        [inimg, u, px, py, tau, weight, nm, error, err_prev, err_init, eps, i, m_itr])

    return u


def mean_shift(input_x, n_updates=-1, window_radius=0.05, bin_seed=True):
    x1 = tf.expand_dims(tf.transpose(input_x), 0)
    x2 = tf.expand_dims(input_x, 0)
    init_c = input_x
    if bin_seed:
        bin_tensor(init_c, window_s=window_radius)
    sbs_c = tf.TensorArray(dtype=tf.float32, size=10000, infer_shape=False)
    sbs_c = sbs_c.write(0, init_c)

    def _mean_shift_step(c):
        c = tf.expand_dims(c, 2)
        Y = tf.reduce_sum(tf.pow((c - x1) / window_radius, 2), axis=1)
        gY = tf.exp(-Y)
        num = tf.reduce_sum(tf.expand_dims(gY, 2) * x2, axis=1)
        denom = tf.reduce_sum(gY, axis=1, keep_dims=True)
        c = num / denom
        return c

    if n_updates > 0:
        for i in range(n_updates):
            c = _mean_shift_step(init_c)
            sbs_c = sbs_c.write(i + 1, c)
    else:
        def _mean_shift(i, c, sbs_c, max_diff):
            new_c = _mean_shift_step(c)
            max_diff = tf.reshape(tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.pow(new_c - c, 2), axis=1))), [])
            sbs_c = sbs_c.write(i + 1, new_c)
            return i + 1, new_c, sbs_c, max_diff

        def _cond(i, c, sbs_c, max_diff):
            return max_diff > 1e-5

        n_updates, c, sbs_c, _ = tf.while_loop(cond=_cond,
                                               body=_mean_shift,
                                               loop_vars=(tf.constant(0), init_c, sbs_c, tf.constant(1e10)))

        #n_updates = tf.Print(n_updates, [n_updates])

    c = tf.reshape(bin_tensor(c, window_s=window_radius), [-1])

    return c, sbs_c.gather(tf.range(n_updates + 1))


def bin_tensor(values, window_s):
    val_range = [tf.reduce_min(values), tf.reduce_max(values)]
    n_bins = tf.cast((val_range[0] - val_range[1]) / window_s, tf.int32)
    all_bins = tf.range(val_range[0] + window_s / 2, val_range[1], window_s)
    ind = tf.histogram_fixed_width_bins(values=values, value_range=val_range, nbins=n_bins)
    counts = tf.bincount(ind)
    bidx = tf.reshape(tf.where(tf.greater(counts, 0)), (-1, 1))
    return tf.gather(all_bins, bidx)


def reshape_clusters_to_cluster_number(clusters, n_clusters):

    def _inner_func(_clusters, n_clusters):
        _clusters = tf.concat([tf.convert_to_tensor([0.0]), _clusters], axis=0)
        _clusters = tf.concat([_clusters, tf.convert_to_tensor([1.0])], axis=0)

        def _lcond(__clusters, n_clusters):
            return tf.not_equal(tf.shape(__clusters)[0], n_clusters + 2)

        def _remove(__clusters, n_clusters):
            def _rloop(__clusters, n_clusters):
                diffs = __clusters[2:] - __clusters[:-2]
                m_idx = tf.argmin(diffs)
                return [tf.concat([__clusters[:m_idx + 1], __clusters[m_idx + 2:]], axis=0), n_clusters]
            return tf.while_loop(cond=_lcond, body=_rloop, loop_vars=(__clusters, n_clusters))[0]

        def _insert(__clusters, n_clusters):
            def _iloop(__clusters, n_clusters):
                diffs = __clusters[1:] - __clusters[:-1]
                m_idx = tf.argmin(diffs)
                return [tf.concat([__clusters[:m_idx + 1], tf.convert_to_tensor([-100.0]), __clusters[m_idx + 1:]],
                                  axis=0), n_clusters]
            return tf.while_loop(cond=_lcond, body=_iloop, loop_vars=(__clusters, n_clusters))[0]

        return tf.cond(tf.greater(tf.shape(_clusters)[0], n_clusters),
                       lambda: _remove(_clusters, n_clusters),
                       lambda: _insert(_clusters, n_clusters))

    return tf.cond(tf.equal(tf.shape(clusters)[0], tf.constant(n_clusters)),
                   lambda: _inner_func(clusters, n_clusters),
                   lambda: (clusters))


def get_kmeans(img, clusters_n, iteration_n):
    points = tf.reshape(img, [tf.shape(img)[0] * tf.shape(img)[1] * tf.shape(img)[2], 1])
    points_expanded = tf.expand_dims(points, 0)
    centroids = tf.random.uniform([clusters_n, 1],
                                  minval=tf.math.reduce_min(points_expanded),
                                  maxval=tf.math.reduce_max(points_expanded))
    #np.random.uniform(0, 10, (clusters_n, 1)).astype()
    #points_expanded = tf.expand_dims(img, 0)
    for step in range(iteration_n):
        centroids_expanded = tf.expand_dims(centroids, 1)
        a = tf.subtract(points_expanded, centroids_expanded)
        b = tf.square(a)
        distances = tf.reduce_sum(b, 2)
        assignments = tf.argmin(distances, 0)

        means = []
        for c in range(clusters_n):
            eq = tf.equal(assignments, c)
            eqw = tf.where(eq)
            eqwr = tf.reshape(eqw, [1, -1])
            eqwrsl = tf.gather(points, eqwr)
            mean = tf.reduce_mean(eqwrsl, reduction_indices=[1])
            means.append(mean)
            #means.append(tf.reduce_mean)
            # tf.gather(points,
            #         tf.reshape(
            #            tf.where(
            #              tf.equal(assignments, c)
            #           ), [1, -1])
            #     ), reduction_indices=[1]))

        new_centroids = tf.concat(means, 0)
        centroids = new_centroids
        #update_centroids = tf.assign(centroids, new_centroids)

    centroids_expanded = tf.expand_dims(centroids, 1)
    centroids_expanded = tf.transpose(centroids_expanded)
    centroids_expanded = tf.reverse(tf.math.top_k(centroids_expanded[0], k=clusters_n, sorted=True).values, [-1])
    centroids_expanded = tf.expand_dims(centroids_expanded, 0)
    distances = tf.square(tf.subtract(img, centroids_expanded))
    assignments = tf.argmin(distances, axis=2, output_type=tf.int32)
    assignments = tf.cast(assignments, tf.float32)

    return assignments


def convert_8bit_image_to_one_hot(image, depth=255):
    """
    Creates a one hot tensor of a given image


    :param image: input image
    :param depth: depth of the one hot tensor default =255 (8bit image)
    :returns: One hot Tensor of depth = depth:
    """
    image = tf.reshape(image, [tf.shape(image)[0], tf.shape(image)[1]])
    if not image.dtype == tf.uint8:
        image = tf.cast(image, tf.uint8)
    if depth != 255:
        image = tf.truediv(image, tf.cast(255, tf.uint8))
        image = tf.scalar_mul(depth, image)
        if not image.dtype == tf.uint8:
            image = tf.cast(image, tf.uint8)
    one_hot = tf.one_hot(image, depth)
    return one_hot


def to_one_hot(image, depth):
    """
    Creates a one hot tensor of a given image


    :param image: input image
    :param depth: depth of the one hot tensor default =255 (8bit image)
    :returns: One hot Tensor of depth = depth:
    """
    if depth == 1:
        mask = tf.greater(image, 0.)
        image = tf.where(mask, tf.ones_like(image), image)
        return image
    if depth == 2: # hole tumor and every thing else
        mask = tf.greater(image, 0.)
        image = tf.where(mask, tf.ones_like(image), image)
    image = tf.reshape(image, [tf.shape(image)[0], tf.shape(image)[1]])
    if not image.dtype == tf.uint8:
        image = tf.cast(image, tf.uint8)
    one_hot = tf.one_hot(image, depth)
    return one_hot


def normalize_and_zero_center_tensor(tensor, max, new_max, normalize_std):
    """
    Creates a one hot tensor of a given image


    :param tensor: input tensor of shape [?, ?, ?, ?]
    :param max: max value of input image as it could be
    :param new_max: new max which the image is normailzed to
    :param normalize_std: True if std should be normalized
    :returns: One hot Tensor of depth = depth:
    """
    if max == new_max:
        normal = tensor
    else:
        normal = tf.math.divide(tensor, tf.constant(max))
        normal = tf.math.multiply(normal, tf.constant(new_max))
    # intensity normalize
    if normalize_std:
        mean, var = tf.nn.moments(normal, axes=[0, 1, 2])
        out = tf.math.divide((normal-mean), tf.math.sqrt(var))
    else:
        mean = tf.reduce_mean(normal)
        out = normal - mean
    #set black elemts to random
    #zero = tf.constant(0, dtype=tf.float32)
    #where_zero = tf.equal(out, zero)
    #out = tf.where(where_zero > 0, tf.random_uniform(out.shape, -0.5, 0.5, dtype=tf.float32, seed=0), out)
    #out = tf.math.divide(tensor, tf.reduce_max(out))
    return out



