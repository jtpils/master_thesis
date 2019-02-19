import numpy as np
from lidar_data_functions import *
from PIL import Image
import time
import matplotlib.pyplot as plt

path_to_ply = '/Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/_out_Town03_190207_18/pc/173504.ply'
path_to_csv = '/Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/_out_Town03_190207_18/Town03_190207_18.csv'


# Load data:
pc, global_lidar_coordinates = load_data(path_to_ply, path_to_csv)
print('Shape of raw pointcloud: ',np.shape(pc))


'''
plt.figure(1)
# PLOT RAW POINT CLOUD:
x = pc[:, 0]
y = pc[:, 1]
plt.plot(x, y, '.')
plt.title('Raw point cloud')
plt.show()
'''

# Trim point cloud:
trimmed_pc = trim_pointcloud(pc, range=30, roof=10, floor=0.1)
print('Shape of trimmed pointcloud: ', np.shape(trimmed_pc))

'''# PLOT TRIMMED POINTCLOUD:
x = trimmed_pc[:, 0]
y = trimmed_pc[:, 1]
plt.plot(x, y, 'r.')
plt.title('Trimmed pointcloud')
plt.show()'''

'''
pc_image = discretize_pointcloud(trimmed_pc)

# NORMALIZE THE BEV IMAGE
max_value = np.max(pc_image[1, :, :])
print('Max evaluation: ', max_value)
scale = 255/max_value
pc_image[1, :, :] = pc_image[1, :, :] * scale
print('Largest pixel value (should be 255) : ', np.max(pc_image[1, :, :]))



# PLOT THE MAX EVALUATION BEV IMAGE
img = Image.fromarray(pc_image[1, :, :])
img.show()

# Save the channels in pc_image as png files in a folder
# array_to_png(pc_image)
'''