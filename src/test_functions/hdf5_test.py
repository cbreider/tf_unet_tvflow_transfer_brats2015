import h5py
import numpy as np

filename = '/home/christian/Projects/Lab_SS2019/caffe-tensorflow/2d_cell_net_v0.modeldef.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = f['model_prototxt']
#txt = open('test.txt', 'w')
#txt.write(','.join(repr(item) for item in data))
#data.tofile("my_data.txt")
#print(data[:])
print("Keys: %s" % data.keys())

d = np.array(data)
np.savetxt('test.txt', d)
a = 0