from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import src.utilities.tf_utils as tfu
import copy as copy
import src.test_functions.np_tv_denoise_test as nptv
from src.utilities.path_utils import DataPaths
from src.utilities.split_utilities import TrainingDataset
import src.utilities.enum_params as enp
import os
import src.utilities.np_data_process_utils as dutils
import time
import configuration as config
import  src.utilities.io_utils as iou


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


tv_tau = 0.125
tv_weight = 0.1
tv_eps = 0.00001
tv_m_itr = 200
km_m_itr = 100
n_clusters = 10


def eval_tf_smooth_and_cluster():
    img = np.array(plt.imread(
        "/home/christian/Projects/Lab_SS2019/dataset/2d_slices/png/raw/train/HGG/brats_2013_pat0003_1/VSD.Brain.XX.O.MR_Flair.54524/VSD.Brain.XX.O.MR_Flair.54524_100.png"))
    # img = np.array(plt.imread("../../../dataset/2d_slices/png/raw/train/HGG/brats_2013_pat0006_1/VSD.Brain.XX.O.MR_T2.54545/VSD.Brain.XX.O.MR_T2.54545_85.png"))
    img = np.reshape(img, (240, 240, 1))

    # run np tv smoothing
    tv_img = nptv.tv_denoise(copy.deepcopy(img), tau=tv_tau, weight=tv_weight, eps=tv_eps, num_iter_max=tv_m_itr)
    tv_img = np.array(np.reshape(tv_img, (240, 240)))

    # run sklearn k-means clustering
    tv_flat = tv_img.reshape((-1, 1))
    km_def = KMeans(n_clusters=n_clusters, tol=0.00000001, max_iter=km_m_itr).fit(tv_flat)
    km_labels = km_def.labels_
    img_km_cl = np.reshape(km_labels, (240,240))

    #create tf pipeline for tv smoothing and clustering
    pl = tf.placeholder(tf.float32, shape=[240, 240, 1])
    tv_cluster = tfu.get_tv_smoothed_and_kmeans_clustering(pl,
                                                           nr_img=2, tv_tau=tv_tau, tv_weight=tv_weight, tv_eps=tv_eps,
                                                           tv_m_itr=tv_m_itr, km_cluster_n=n_clusters, km_itr_n=km_m_itr)
    # run tf
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        tv_smoothed, clustering= sess.run(tv_cluster, feed_dict={pl: img})
        img_tv_tf = np.array(np.reshape(tv_smoothed, (240, 240)))
        img_km_tf = np.array(np.reshape(clustering, (240, 240)))
        X = np.reshape(img, (240, 240))
        plt.matshow(X, "orig")
        plt.matshow(img_tv_tf, "tv_tf")
        plt.matshow(tv_img, "tv_np")
        plt.matshow(img_km_tf, "km tf")
        plt.matshow(img_km_cl, "km_np")
        plt.show()


def eval_binning_and_meanshift():
    img = np.array(plt.imread(
        "/home/christian/Projects/Lab_SS2019/dataset/2d_slices/png/raw/train/HGG/brats_2013_pat0003_1/VSD.Brain.XX.O.MR_Flair.54524/VSD.Brain.XX.O.MR_Flair.54524_100.png"))
    # img = np.array(plt.imread("../../../dataset/2d_slices/png/raw/train/HGG/brats_2013_pat0006_1/VSD.Brain.XX.O.MR_T2.54545/VSD.Brain.XX.O.MR_T2.54545_85.png"))
    img = np.reshape(img, (240, 240, 1))


    # run sklearn k-means clustering

    #create tf pipeline for tv smoothing and clustering
    pl = tf.placeholder(tf.float32, shape=[240, 240, 1])
    tv = tfu.get_tv_smoothed(pl, tau=tv_tau, weight=tv_weight, eps=tv_eps, m_itr=tv_m_itr)
    tv = tfu.normalize_and_zero_center_tensor(tv, max=255.0, new_max=1.0, normalize_std=True)
    binned = tfu.get_fixed_bin_clustering(image=pl, n_bins=10)
    ms = tfu.get_meanshift_clustering(image=pl, ms_itr=50, win_r=0.03, n_clusters=10, bin_seeding=True)
    # run tf
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        t_st = time.time()
        tv_smoothed = sess.run(tv, feed_dict={pl: img})
        t_tv = time.time()
        binning = sess.run(binned, feed_dict={pl: tv_smoothed})
        t_b = time.time()
        ms_clustering = sess.run(ms, feed_dict={pl: tv_smoothed})
        t_ms = time.time()
        img_tv_tf = np.array(np.reshape(tv_smoothed, (240, 240)))
        img_bn = np.array(np.reshape(binning, (240, 240)))
        img_ms = np.array(np.reshape(ms_clustering, (240, 240)))
        print("TV: {}  Bin: {}  MS: {}".format(t_tv-t_st, t_b-t_tv, t_ms-t_b))
        X = np.reshape(img, (240, 240))
        plt.matshow(X, "orig")
        plt.matshow(img_tv_tf, "img_tv")
        plt.matshow(img_bn, "imh_bn")
        plt.matshow(img_ms, "img_ms")
        plt.show()


def compare_clusterings():
    if not os.path.exists("test_out"):
        os.makedirs("test_out")
    data_paths = DataPaths(data_path="/home/christian/Projects/LabSS2019/dataset", mode="TVFLOW")
    data_paths.load_data_paths(mkdirs=True)
    file_paths = TrainingDataset(paths=data_paths, mode=enp.TrainingModes.TVFLOW_REGRESSION, data_config=config.DataParams,
                                 new_split=True,
                                 is_five_fold=False,
                                 five_fold_idx=0, nr_of_folds=config.DataParams.nr_k_folds,
                                 k_fold_nr_val_samples=config.DataParams.k_fold_nr_val_samples)

    all_imgs = []
    all_tvs = []
    cl_img_flat = []
    i = 0
    for k, path in file_paths.train_paths.items():
        img = np.array(plt.imread(path[0]))
        if np.max(img) == 0.0:
            continue
        all_imgs.append(img)
        img = np.reshape(img, (240, 240, 1))
        tv_img = nptv.tv_denoise(copy.deepcopy(img), tau=tv_tau, weight=tv_weight, eps=tv_eps, num_iter_max=tv_m_itr)
        tv_img = np.array(np.reshape(tv_img, (240, 240)))
        all_tvs.append(tv_img)
        if int(np.random.uniform(1, 10))%5 == 0:
            cl_img_flat.append(tv_img.reshape((-1, 1)))
        i += 1
        print(i)
        if i == 200:
            break
    print("TV done!")
    cl_img_flat = np.array(cl_img_flat)
    cl_img_flat = cl_img_flat.reshape((-1, 1))
    km_def = KMeans(n_clusters=n_clusters, tol=0.00000001,
                    max_iter=km_m_itr, precompute_distances=True, n_jobs=6).fit(cl_img_flat)
    km_cl_cen = np.sort(km_def.cluster_centers_.reshape(n_clusters))
    bin_size = 1.0 / float(n_clusters)
    hard_cl_cen = np.array([(float(c) + 0.5) * bin_size for c in range(0, n_clusters)])
    hard_cl_cen = hard_cl_cen
    print("K-means done!")
    for i in range(len(all_tvs)):
        in_arr = np.repeat(np.expand_dims(all_tvs[i], 2), n_clusters, axis=2)
        # get hard bin assignmnets:
        dist = np.subtract(in_arr, hard_cl_cen)
        dist = np.square(dist)
        hard_assign = np.argmin(dist, axis=2)

        #get pr-clustered k-means assignments
        dist = np.subtract(in_arr, km_cl_cen)
        dist = np.square(dist)
        km_assign = np.argmin(dist, axis=2)

        tv_flat = all_tvs[i].reshape((-1, 1))

        #get new km assignmnets
        km_new_def = KMeans(n_clusters=n_clusters, tol=0.00000001,
                        max_iter=km_m_itr, precompute_distances=True, n_jobs=6).fit(tv_flat)
        km_new_cl_cen = np.sort(km_new_def.cluster_centers_.reshape(n_clusters))
        dist = np.subtract(in_arr, km_new_cl_cen)
        dist = np.square(dist)
        km_new_assign = np.argmin(dist, axis=2)

        #get mean shift clusteirng
        bandwidth = estimate_bandwidth(tv_flat, quantile=0.05, n_samples=500)
        ms_def = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(tv_flat)
        ms_clusters = ms_def.cluster_centers_.reshape(ms_def.cluster_centers_.shape[0])
        ms_clusters = np.sort(ms_clusters)
        if ms_clusters.shape[0] != n_clusters:
            ms_clusters = np.insert(ms_clusters, 0, 0.0)
            ms_clusters = np.insert(ms_clusters, ms_clusters.shape[0], 1.0)
            # if number of clusters is greater than it should be delete elements with smallest distance to its neighbor
            if ms_clusters.shape[0] > n_clusters+2:
                while not (ms_clusters.shape[0] == n_clusters+2):
                    # get for each elememnt its distance to ist neighbor
                    diffs = ms_clusters[2:] - ms_clusters[:-2]
                    m_idx = np.argmin(diffs)
                    ms_clusters = np.concatenate([ms_clusters[:m_idx + 1], ms_clusters[m_idx + 2:]])#np.delete(ms_clusters, m_idx + 1)

            elif ms_clusters.shape[0] < n_clusters+2:
                while not (ms_clusters.shape[0] == n_clusters+2):
                    diffs = ms_clusters[1:] - ms_clusters[:-1]
                    m_idx = np.argmax(diffs)
                    ms_clusters = np.concatenate([ms_clusters[:m_idx + 1], [-100.0], ms_clusters[m_idx + 1:]])#np.insert(ms_clusters, m_idx + 1, -100.0)
            ms_clusters = ms_clusters[1:-1]
        else:
            ms_clusters = ms_clusters

        dist = np.subtract(in_arr, ms_clusters)
        dist = np.square(dist)
        ms_assign = np.argmin(dist, axis=2)

        img = np.concatenate((all_imgs[i].astype(np.float) / all_imgs[i].max() * 255.0,
                              all_tvs[i].astype(np.float) / all_tvs[i].max() * 255.0,
                              hard_assign.astype(np.float) / float(n_clusters) * 255.0,
                              km_assign.astype(np.float) / float(n_clusters) * 255.0,
                              km_new_assign.astype(np.float) / float(n_clusters) * 255.0,
                              ms_assign.astype(np.float) / float(n_clusters) * 255.0),
                             axis=1)
        iou.save_image(img, "test_out/{}.jpg".format(i))
        print("Clustered {} of {}".format(i, len(all_tvs)))




if __name__ == "__main__":
    #eval_tf_smooth_and_cluster()
    compare_clusterings()
    #nptv.test_anisotropic_diffiusion_smoothing()
    #eval_binning_and_meanshift()