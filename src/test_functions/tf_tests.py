import tensorflow as tf
import  numpy as np
import src.utils.tf_utils as tfu
import matplotlib.pyplot as plt

img = np.array(plt.imread(
    "/home/christian/Projects/Lab_SS2019/dataset/2d_slices/png/raw/train/HGG/brats_2013_pat0003_1/VSD.Brain.XX.O.MR_Flair.54524/VSD.Brain.XX.O.MR_Flair.54524_100.png"))
img = np.reshape(img, (240, 240, 1))
pl = tf.placeholder(tf.float32, shape=[240, 240, 1])
tv = tfu.get_tv_smoothed(pl, tau=0.125, weight=0.1, eps=0.00001, m_itr=200)
tv_r = tf.reshape(tv, (-1, 1))
bins = tfu.bin_tensor(tv_r, window_s=0.02)

ms_r = tfu.mean_shift(tv_r)
#op = tfu.reshape_clusters_to_cluster_number(ms_r, 10)
op = tfu.get_tv_smoothed_and_meanshift_clusterd_one_hot(image=img, tv_tau=0.125, tv_weight=0.1, tv_eps=0.00001, tv_m_itr=200, ms_itr=-1, win_r=0.02,
                                                   n_clusters=10)
with tf.Session() as sess:
    t, tr, b, ms = sess.run([tv, tv_r, bins, ms_r], feed_dict={pl: img})
    s = sess.run(tf.shape(ms)[0])
    op= sess.run(op, feed_dict={ms_r: ms})

S = tf.placeholder(dtype=tf.float32, shape=[11])
DIFF = S[1:]-S[:-1]
DIFF2 = S[2:]-S[:-2]
IDX_MIN = tf.argmin(DIFF2)
IDX_MAX = tf.argmax(DIFF)
s11 = S[:IDX_MIN+1]
s12 = S[IDX_MIN+2:]
s21 = S[:IDX_MAX+1]
s22 = S[IDX_MAX+1:]
S1 = tf.concat([s11, s12], axis=0)
S2 = tf.concat([s21, tf.convert_to_tensor([-100.0]), s22], axis=0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    s, diff, diff2,  idx_min, idx_max, s1, s2 = sess.run([S, DIFF, DIFF2, IDX_MIN, IDX_MAX, S1, S2],
                                                 feed_dict={S: np.array([0.0, 1.0, 3.0, 6.0, 7.0, 7.5, 9.0, 11.0, 14.2, 14.5, 15.0])})
    a = 0


S1 = tf.stack(S[:4-1])
A = tf.convert_to_tensor([[0, 0, 1, 2, 0, 0],
     [0, 0, 1, 2, 5, 0],
     [0, 3, 1, 2, 0, 0],
     [0, 0, 1, 2, 3, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]])
A = tf.reshape(A, [6, 6, 1])

zero = tf.constant(0, dtype=tf.int32)
where = tf.not_equal(A, zero)
indices = tf.where(where)
min_y = tf.reduce_min(indices[:, 0])
min_x = tf.reduce_min(indices[:, 1])
max_y = tf.reduce_max(indices[:, 0])
max_x = tf.reduce_max(indices[:, 1])
height = tf.math.add(tf.math.subtract(max_y, min_y), tf.convert_to_tensor(1, dtype=tf.int64))
width = tf.math.add(tf.math.subtract(max_x, min_x), tf.convert_to_tensor(1, dtype=tf.int64))
crop = tf.image.crop_to_bounding_box(A, min_y, min_x, height, width)
resize = tf.image.resize_images(crop, [6, 6], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
resh_crop = tf.reshape(resize, [6, 6])
with tf.Session() as sess:
 t = sess.run(A)
 x = sess.run(tf.shape(t))
 out = sess.run(indices)
 np_ou = np.array(out)
 tf.print(out)
 minX = sess.run(min_x)
 minY = sess.run(min_y)
 maxX = sess.run(height)
 maxY = sess.run(width)
 cropped = sess.run(resh_crop)
 #tf.print(cropped)
 np_crop = np.array(cropped)
 #print(np_crop)

 f = 0