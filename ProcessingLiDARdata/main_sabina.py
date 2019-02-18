import numpy as np
from lidar_data_functions import *
from PIL import Image
import time
import matplotlib.pyplot as plt
import os




path_to_ply = '/home/master04/Desktop/_out/_out_Town02_190208_1/pc/004179.ply'
path_to_csv = '/home/master04/Desktop/_out/_out_Town02_190208_1/Town02_190208_1.csv'


point_cloud, global_coordinates = load_data(path_to_ply, path_to_csv)
# testing rotate and translate

#discretized_pc_not_rot = discretize_pointcloud(point_cloud, spatial_resolution=0.05)

#show img

# NORMALIZE THE BEV IMAGE
#for channel in range(np.shape(discretized_pc_not_rot)[0]):
#    max_value = np.max(discretized_pc_not_rot[channel, :, :])
#    print('Max max_value inarray_to_png: ', max_value)
#    # avoid division with 0
#    if max_value == 0:
#        max_value = 1
#    scale = 255/max_value
#    discretized_pc_not_rot[channel, :, :] = discretized_pc_not_rot[channel, :, :] * scale
#    print('Largest pixel value (should be 255) : ', np.max(discretized_pc_not_rot[channel, :, :]))

# show img
#img = Image.fromarray(discretized_pc_not_rot[1, :, :])
#img.show()

#ax1 = plt.subplot(121)
#x = point_cloud[:, 0]
#y = point_cloud[:, 1]
#ax1.plot(x, y, 'r.')
#ax1.axis('equal')
#ax1.set_title('raw point cloud')
#ax1.show()

number_of_points = len(point_cloud)  # number of points in the pointcloud


#New
yaw = np.radians(90)
c, s = np.cos(yaw), np.sin(yaw)
Rz = np.array(([c, -s, 0], [s, c, 0], [0, 0, 1]))  # Rotation matrix
rotated_pointcloud_new = np.matmul(Rz, np.transpose(point_cloud)) # rotate each vector with coordinates, transpose to get dimensions correctly
rotated_pointcloud_new = np.transpose(np.reshape(rotated_pointcloud_new, (3, number_of_points)))  # reshape and transpose back
print(global_coordinates[0, :3])
translated_pc_new = rotated_pointcloud_new + global_coordinates[0, :3]
#discretized_pc_new = discretize_pointcloud(translated_pc_new, spatial_resolution=0.05)

#ax2 = plt.subplot(122)
#x = translated_pc_new[:, 0]
#y = translated_pc_new[:, 1]
#ax2.plot(x, y, 'r.')
#ax2.axis('equal')
#ax2.set_title('translated and rotated pc')
#plt.show()


#show img

# NORMALIZE THE BEV IMAGE
#for channel in range(np.shape(discretized_pc_new)[0]):
#    max_value = np.max(discretized_pc_new[channel, :, :])
#    print('Max max_value inarray_to_png: ', max_value)
#    # avoid division with 0
#    if max_value == 0:
#        max_value = 1
#    scale = 255/max_value
#    discretized_pc_new[channel, :, :] = discretized_pc_new[channel, :, :] * scale
#    print('Largest pixel value (should be 255) : ', np.max(discretized_pc_new[channel, :, :]))

## show img
#img = Image.fromarray(discretized_pc_new[1, :, :])
#img.show()
'''

#old
yaw = np.deg2rad(90)  # convert yaw in degrees to radians
c, s = np.cos(yaw), np.sin(yaw)
Rz = np.array(([c, -s, 0], [s, c, 0], [0, 0, 1]))  # Rotation matrix
rotated_pointcloud = Rz @ np.transpose(point_cloud) # rotate each vector with coordinates, transpose to get dimensions correctly
rotated_pointcloud = np.transpose(np.reshape(rotated_pointcloud, (3, number_of_points)))  # reshape and transpose back
print(global_coordinates[0, :3])
translated_pc_new = rotated_pointcloud + global_coordinates[0, :3]

print(point_cloud[:30])
print(global_coordinates[0, :])
print(global_coordinates[0, 3])

discretized_pc_old = discretize_pointcloud(translated_pc_new, spatial_resolution=0.05)

#show img

# NORMALIZE THE BEV IMAGE
for channel in range(np.shape(discretized_pc_old)[0]):
    max_value = np.max(discretized_pc_old[channel, :, :])
    print('Max max_value inarray_to_png: ', max_value)
    # avoid division with 0
    if max_value == 0:
        max_value = 1
    scale = 255/max_value
    discretized_pc_old[channel, :, :] = discretized_pc_old[channel, :, :] * scale
    print('Largest pixel value (should be 255) : ', np.max(discretized_pc_old[channel, :, :]))

# show img
img = Image.fromarray(discretized_pc_old[1, :, :])
img.show()






# PLOT TRIMMED POINTCLOUD:
x = trimmed_pc[:, 0]
y = trimmed_pc[:, 1]
plt.plot(x, y, 'r.')
plt.axis('equal')
plt.title('Trimmed pointcloud')
plt.show()


rigid_trans = random_rigid_transformation(10, 10)
print('rigid trans: ', rigid_trans)
train_pc = training_sample_rotation_translation(trimmed_pc, rigid_trans)

x_train = train_pc[:, 0]
y_train = train_pc[:, 1]
plt.plot(x, y, 'r.', x_train, y_train, 'b.')
plt.axis('equal')
plt.show() '''
