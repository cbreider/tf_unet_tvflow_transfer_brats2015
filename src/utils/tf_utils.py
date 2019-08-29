"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019
"""


import tensorflow as tf


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



def crop_images_to_to_non_zero(scan, ground_truth, size):
	"""
	crops input and gt images to bounding box of non zero area of gt image


	:param scan: input image
	:param ground_truth: ground_truth image
	:returns:cobmbined padded,cropped and flipped images
	"""
	# HACK check if gt is completly zero then return orginal images
	total = tf.reduce_sum(tf.abs(ground_truth))
	is_all_zero = tf.equal(total, 0)
	return tf.cond(is_all_zero, lambda: (scan, ground_truth), lambda: crop_non_zero_internal(scan, ground_truth, size))


def crop_non_zero_internal(scan, ground_truth, out_size):
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
	resize_in = tf.image.resize_images(crop_in, out_size,
									   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	resize_gt = tf.image.resize_images(crop_gt, out_size,
									   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	return tf.cast(resize_in, tf.float32), tf.cast(resize_gt, tf.float32)


def load_png_image(filename, nr_channels, img_size, data_type=tf.float32):
	"""
	Loads a png imges within a TF pipeline

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



