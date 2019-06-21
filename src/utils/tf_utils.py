"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019
"""


import tensorflow as tf
import src.configuration as conf


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
	combined_pad = tf.image.pad_to_bounding_box(
		combined, 0, 0,
		tf.maximum(conf.DataParams.image_size[0], image_shape[0]),
		tf.maximum(conf.DataParams.image_size[1], image_shape[1]))
	last_label_dim = tf.shape(ground_truth)[-1]
	last_image_dim = tf.shape(scan)[-1]
	combined_crop = tf.random_crop(value=combined_pad,
								   size=tf.concat([conf.image_size, [last_label_dim + last_image_dim]], axis=0))
	combined_flip = tf.image.random_flip_left_right(combined_crop)
	# mean = tf.metrics.mean(scan)
	# combined_centered = combined_flip - mean
	return combined_flip[:, :, :last_image_dim], combined_flip[:, :, last_image_dim:]


def load_png_image(filename, data_type=tf.float32):
	"""
	Loads a png imges within a TF pipeline

	:param filename: use scale images from tvflow as gt instead of smoothed images
	:param data_type: data type in which the image is casted
	:returns: image of dtype = data_type
	"""
	img_string = tf.read_file(filename)
	img_decoded = tf.image.decode_png(img_string, channels=conf.DataParams.nr_of_channels)
	img_resized = tf.image.resize_images(img_decoded, size=conf.DataParams.image_size)
	img = tf.cast(img_resized, data_type)
	return img


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


