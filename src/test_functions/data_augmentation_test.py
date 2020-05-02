import tensorflow as tf
import  numpy as np
import src.utilities.tf_utils as tfu
import src.utilities.np_data_process_utils as npu
import src.utilities.io_utils as iou
import matplotlib.pyplot as plt
from PIL import Image

img = np.array(Image.open(
    "/home/christian/Projects/Lab_SS2019/dataset/2d_slices/png/raw/train/HGG/brats_2013_pat0006_1/VSD.Brain.XX.O.MR_T2.54545/VSD.Brain.XX.O.MR_T2.54545_82.png"))

gt = np.array(Image.open(
    "/home/christian/Projects/Lab_SS2019/dataset/2d_slices/png/raw/train/HGG/brats_2013_pat0006_1/VSD.Brain_3more.XX.O.OT.54547/VSD.Brain_3more.XX.O.OT.54547_82.png"))

gt = np.expand_dims(gt, axis=2)
img = np.expand_dims(img, axis=2)

in_img = tf.placeholder(tf.float32, shape=[240, 240, 1])
gt_img = tf.placeholder(tf.float32, shape=[240, 240, 1])


out_img = tfu.distort_imgs(in_img, gt_img, params=[[2, 3, 3], 25.0, 0.8])
gt_img_o1 = tf.reshape(out_img[1], [tf.shape(out_img[1])[0], tf.shape(out_img[1])[1]])
gt_img_o = tf.one_hot(tf.cast(gt_img_o1, tf.int32), 5)

with tf.Session() as sess:
     [img_dis, gt_o] = sess.run([out_img, gt_img_o], feed_dict={in_img: img, gt_img: gt})
     img_dis = (img_dis[0] - img_dis[0].min()) / img_dis[0].max() * 255.0
     img_dis = npu.to_rgb(img_dis)
     gt_dis = npu.one_hot_to_rgb(gt_o, img_dis)
     iou.save_image(img_dis, "test_in.jpg")
     iou.save_image(gt_dis, "test_gt.jpg")

