import tensorflow as tf

import h5py
import numpy as np

filename = '/home/christian/Projects/Lab_SS2019/caffe-tensorflow/2d_cell_net_v0.caffemodel.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
all_data = f['data']
print("Keys: %s" % all_data.keys())

data_conv2 = all_data['']
#txt = open('test.txt', 'w')
#txt.write(','.join(repr(item) for item in data))
#data.tofile("my_data.txt")
#print(data[:])
a = 0


tf.reset_default_graph()

# Create some variables.

# Add ops to save and restore all the variables.
loaded_graph = tf.Graph()
# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session(graph=loaded_graph) as sess:
  # Restore variables from disk.
  loader = tf.train.import_meta_graph('/home/christian/Projects/Lab_SS2019/unet_brats2015/pretrained_model/model-8500.meta')
  loader.restore(sess, "/home/christian/Projects/Lab_SS2019/unet_brats2015/pretrained_model/model-8500")
  print("Model restored.")