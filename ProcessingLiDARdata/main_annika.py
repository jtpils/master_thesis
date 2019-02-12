import numpy as np
from lidar_data_functions import *
from PIL import Image
import matplotlib.pyplot as plt

#path_to_ply = '/home/master04/Desktop/_out/_out_Town02_190208_1/pc/004179.ply'
#path_to_csv = '/home/master04/Desktop/_out/_out_Town02_190208_1/Town02_190208_1.csv'

# Load data:
#pc, global_lidar_coordinates = load_data(path_to_ply, path_to_csv)
#print('Shape of raw pointcloud: ',np.shape(pc))

# PLOT RAW POINT CLOUD:
'''x = pc[:, 0]
y = pc[:, 1]
plt.plot(x, y, 'ro')
plt.title('Raw point cloud')
plt.show()

# Test array:
pc = np.array([[-10,-10, 0], [-15, 3, 3], [26, 2, 3], [1, 2, 3], [10, 28, 9], [4, 1, -4], [4, 1, 5], [11, 9, 1], [12, 1, 10],[4, 8, 10], [1, 2, 3], [5, 7, 9], [25, 9, 1], [23, 3, 10]])
#print('Shape of test array: ', np.shape(pc))
#print(pc[0,:])
#pc = np.array([[-1,-1, 0], [1, 1, 0]])
x = pc[:, 0]
y = pc[:, 1]
#plt.plot(x, y, 'ro')
#plt.title('Raw point cloud')
#plt.show()

print('Shape of raw pointcloud: ',np.shape(pc))


# Trim point cloud:
trimmed_pc = trim_pointcloud(pc)
print('Shape of trimmed pointcloud: ', np.shape(trimmed_pc))
x = trimmed_pc[:, 0]
y = trimmed_pc[:, 1]

# PLOT TRIMMED POINTCLOUD:
x = trimmed_pc[:, 0]
y = trimmed_pc[:, 1]
plt.plot(x, y, 'ro')
plt.title('Trimmed pointcloud')
plt.show()

global_coordinates = [0, 0, 5]
rotated_pc = rotate_pointcloud(trimmed_pc, global_coordinates)
print('shape of rotated pointcloud: ', np.shape(rotated_pc))
x_rot = rotated_pc[:, 0]
y_rot = rotated_pc[:, 1]
plt.plot(x_rot, y_rot, 'ro', x, y, 'bo')
#plt.axis([-2, 2, -2, 2])
plt.title('raw: blue, rotated: red')
plt.show()'''

rigid_trans = random_rigid_transformation(1, 10)
print(rigid_trans)
