from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
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
    nm = np.prod(img.shape[:3])

    i = 0
    while i < num_iter_max:
        u_old = u

        # x and y components of u's gradient
        ux = np.roll(u, -1, axis=2) - u
        uy = np.roll(u, -1, axis=1) - u

        # update the dual variable
        px_new = px + (tau / weight) * ux
        py_new = py + (tau / weight) * uy
        norm_new = np.maximum(1, np.sqrt(px_new ** 2 + py_new ** 2))
        px = px_new / norm_new
        py = py_new / norm_new

        # calculate divergence
        rx = np.roll(px, 1, axis=2)
        ry = np.roll(py, 1, axis=1)
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
weight = 0.3
eps = 0.00001
m_itr = 200

X1 = np.array(plt.imread("/home/christian/Projects/Lab_SS2019/dataset/2d_slices/png/raw/train/HGG/brats_2013_pat0006_1/VSD.Brain.XX.O.MR_T2.54545/VSD.Brain.XX.O.MR_T2.54545_85.png"))
X2 = np.array(plt.imread("/home/christian/Projects/Lab_SS2019/dataset/2d_slices/png/raw/train/HGG/brats_2013_pat0006_1/VSD.Brain.XX.O.MR_T1c.54544/VSD.Brain.XX.O.MR_T1c.54544_85.png"))
X = np.array([X1, X2])
X = np.reshape(X, (2, 240, 240, 1))
t = denoise(copy.deepcopy(X), tau=tau, weight=weight, eps=eps, num_iter_max=m_itr)
out = t - t.min()
out = out / out.max()
img1t = np.array(np.reshape(out[0], (240, 240)))
img2t = np.array(np.reshape(out[1], (240, 240)))


pl = tf.placeholder(tf.float32, shape=[2, 240, 240, 1])
tv_func = tfu.get_tv_smoothed(pl, tau=tau, weight=weight, eps=eps, m_itr=m_itr)
with tf.Session() as sess:

    tf.global_variables_initializer().run()
    out = sess.run(tv_func, feed_dict={pl: X})
    out = out - out.min()
    out = out / out.max()
    img1 = np.array(np.reshape(out[0], (240, 240)))
    img2 = np.array(np.reshape(out[1], (240, 240)))
    plt.matshow(X1)
    plt.matshow(img1)
    plt.matshow(img1t)

    plt.matshow(X2)
    plt.matshow(img2)
    plt.matshow(img2t)
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
