import tensorflow as tf
import  numpy as np
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