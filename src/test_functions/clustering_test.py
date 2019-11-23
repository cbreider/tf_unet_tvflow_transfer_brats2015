from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from scipy import misc
from PIL import Image
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
import tensorflow as tf
import matplotlib.pyplot as plt
import src.utils.tf_utils as tfu
import copy as copy


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


def denoise(img, weight=0.1, tau=0.125, eps=1e-3, num_iter_max=200, ):
    """Perform total-variation denoising on a grayscale image.

    Parameters
    ----------
    img : array
        2-D input data to be de-noised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more de-noising (at
        the expense of fidelity to `img`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    num_iter_max : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : array
        De-noised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """
    u = np.zeros_like(img)
    px = np.zeros_like(img)
    py = np.zeros_like(img)
    nm = np.prod(img.shape[:2])

    i = 0
    while i < num_iter_max:
        u_old = u

        # x and y components of u's gradient
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u

        # update the dual variable
        px_new = px + (tau / weight) * ux
        py_new = py + (tau / weight) * uy
        norm_new = np.maximum(1, np.sqrt(px_new ** 2 + py_new ** 2))
        px = px_new / norm_new
        py = py_new / norm_new

        # calculate divergence
        rx = np.roll(px, 1, axis=1)
        ry = np.roll(py, 1, axis=0)
        div_p = (px - rx) + (py - ry)

        # update image
        u = img + weight * div_p

        # calculate error
        a =np.sqrt(nm)
        error = np.linalg.norm(u - u_old) / np.sqrt(nm)

        if i == 0:
            err_init = error
            err_prev = error
        else:
            # break if error small enough
            if np.abs(err_prev - error) < eps * err_init:
                break
            else:
                e_prev = error

        # don't forget to update iterator
        i += 1

    return u

tau = 0.125
weight = 0.1
eps = 0.00001
m_itr = 200

#x1 = np.array(plt.imread("/home/christian/Projects/LabSS2019/dataset/2d_slices/png/raw/train/HGG/brats_tcia_pat230_0710/VSD.Brain.XX.O.MR_Flair.35788/VSD.Brain.XX.O.MR_Flair.35788_86.png"))
x1 = np.array(plt.imread("../dataset/2d_slices/png/raw/train/HGG/brats_2013_pat0006_1/VSD.Brain.XX.O.MR_T2.54545/VSD.Brain.XX.O.MR_T2.54545_85.png"))
Xs = [x1]
for X in Xs:

    #X2 = np.array(plt.imread("/home/christian/Projects/Lab_SS2019/dataset/2d_slices/png/raw/train/HGG/brats_2013_pat0006_1/VSD.Brain.XX.O.MR_T1c.54544/VSD.Brain.XX.O.MR_T1c.54544_85.png"))
    #X = np.array([X1, X2])
    X = np.reshape(X, (240, 240, 1))
    t = denoise(copy.deepcopy(X), tau=tau, weight=weight, eps=eps, num_iter_max=m_itr)
    out = t - t.min()
    out = out / out.max()
    img1t = np.array(np.reshape(out, (240, 240)))
    tv_flat = img1t.reshape((-1, 1))
    bandwidth = estimate_bandwidth(tv_flat, quantile=0.15, n_samples=500)
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True).fit(tv_flat)
    labels=ms.labels_
    img1_clms = np.reshape(labels, (240,240))

    km = KMeans(n_clusters=7, tol=0.00000001, max_iter=100).fit(tv_flat)
    km_labels = km.labels_
    img1km_cl = np.reshape(km_labels, (240,240))

    #img1_cl = recreate_image(ms.cluster_centers_, cl, 240, 240)
    plt.matshow(img1t, "tv_cost")
    plt.matshow(img1_clms, "ms cl")
    plt.matshow(img1km_cl, "km sk cl")

    #img2t = np.array(np.reshape(out[1], (240, 240)))

    pl = tf.placeholder(tf.float32, shape=[240, 240, 1])
    tv_cluster = tfu.get_tv_smoothed_and_clusterd_one_hot(pl, nr_img=2,
                                                        tv_tau=tau, tv_weight=weight, tv_eps=eps,
                                                        tv_m_itr=m_itr, km_cluster_n=7,
                                                                                  km_itr_n=100)
    #tv_func = tfu.get_tv_smoothed(pl, tau=tau, weight=weight, eps=eps, m_itr=m_itr)
    #kmeans_cluster, assignments = tfu.get_kmeans(tv_func, 6, iteration_n=300)

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        tv_smoothed, clustering= sess.run(tv_cluster, feed_dict={pl: X})
        img1_tv = np.array(np.reshape(tv_smoothed, (240, 240)))
        #img2_tv = np.array(np.reshape(tv_smoothed[1], (240, 240)))
        img1_cl = np.array(np.reshape(clustering, (240, 240)))
        #img2_cl = np.array(np.reshape(clustering[1], (240, 240)))
        X = np.reshape(X, (240, 240))
        plt.matshow(X, "orig")
        plt.matshow(img1_tv, "tv_tf")
        plt.matshow(img1_cl, "km tf")
        #plt.matshow(X2)
        #plt.matshow(img2_tv)
        #plt.matshow(img2_cl)

plt.show()


'''
nz = X[np.nonzero(X)]
nz = nz.reshape((-1, 1))
X = X.reshape((-1, 1))
clustering = KMeans(n_clusters=4, tol=0.00000001, max_iter=1000).fit(X)
#clustering = MeanShift(n_jobs=6).fit(X)
cl = clustering.predict(X)
t = recreate_image(clustering.cluster_centers_, cl, 240, 240)
t *= 255
t = t.astype(np.uint8)
t = t.reshape(240, 240)
img = Image.fromarray(t)
img.show()
'''
'''
img = np.zeros((240, 240))
img[X>0.851] = 1
img[X>1] = 0

img[X<0.48] = 0.5
img[X<0.3] = 0

'''
'''
mask = X.astype(bool)

img = X.astype(float)

# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.img_to_graph(img, mask=mask)

# Take a decreasing function of the gradient: we take it weakly
# dependent from the gradient the segmentation is close to a voronoi
graph.data = np.exp(-graph.data / graph.data.std())

# Force the solver to be arpack, since amg is numerically
# unstable on this example
labels = spectral_clustering(graph, n_clusters=30, eigen_solver='arpack')
label_im = np.full(mask.shape, -1.)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)
plt.show()
'''
