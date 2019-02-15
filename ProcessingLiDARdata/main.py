import numpy as np
from lidar_data_functions import *
from PIL import Image
import time
import matplotlib.pyplot as plt

path_to_ply = '/home/master04/Desktop/_out/_out_Town02_190208_1/pc/004179.ply'
path_to_csv = '/home/master04/Desktop/_out/_out_Town02_190208_1/Town02_190208_1.csv'

# Load data:
pc, global_lidar_coordinates = load_data(path_to_ply, path_to_csv)
print('Shape of raw pointcloud: ',np.shape(pc))


# PLOT RAW POINT CLOUD:
'''x = pc[:, 0]
y = pc[:, 1]
plt.plot(x, y, 'ro')
plt.title('Raw point cloud')
plt.show()'''

# Test array:
#pc = np.array([[-10,-10, 0], [-15, 3, 3], [26, 2, 3], [1, 2, 3], [10, 28, 9], [4, 1, -4], [4, 1, 5], [11, 9, 1], [12, 1, 10],[4, 8, 10], [1, 2, 3], [5, 7, 9], [25, 9, 1], [23, 3, 10]])
#print('Shape of test array: ', np.shape(pc))
#print(pc[0,:])

# Trim point cloud:
trimmed_pc = trim_pointcloud(pc)
print('Shape of trimmed pointcloud: ', np.shape(trimmed_pc))

'''# PLOT TRIMMED POINTCLOUD:
x = trimmed_pc[:, 0]
y = trimmed_pc[:, 1]
plt.plot(x, y, 'r.')
plt.title('Trimmed pointcloud')
plt.show()'''

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

