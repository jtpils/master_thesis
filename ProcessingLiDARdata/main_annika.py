import numpy as np
from lidar_data_functions import *
from matplotlib import pyplot as plt
import pandas as pd


path_to_ply = '/home/master04/Desktop/Ply_files/_out_Town02_190306_1/pc/002198.ply'
path_to_csv = '/home/master04/Desktop/Ply_files/_out_Town02_190306_1/Town02_190306_1.csv'


pc, global_lidar_coordinates = load_data(path_to_ply, path_to_csv)

# trim the point cloud
pc = trim_pointcloud(pc, range=15, roof=20, floor=0)

#plt.plot(pc[:,0], pc[:,1], '.')
#plt.show()

# discretize the sweep with padding
pc = discretize_pointcloud(pc, array_size=600, trim_range=15, spatial_resolution=0.05, padding=False, pad_size=150)

visualize_detections(pc,1)

visualize_detections(pc,2)
plt.show()

#plt.imshow(pc[0,:,:], cmap='gray')
#plt.show()
'''
# Uncomment for visualisation of the sweep and cut_out
layer = 2
max_value = np.max(pc[layer, :, :])
print('Max max_value in array_to_png: ', max_value)

# avoid division with 0
if max_value == 0:
    max_value = 1

scale = 255 / max_value
pc[layer, :, :] = pc[layer, :, :] * scale
print('Largest pixel value (should be 255) : ', np.max(pc[layer, :, :]))


img = Image.fromarray(pc[layer, :, :])
new_img = img.convert("L")
new_img.rotate(180).show()
'''


