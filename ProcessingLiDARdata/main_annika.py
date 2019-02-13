import numpy as np
from lidar_data_functions import *
from PIL import Image
import matplotlib.pyplot as plt

path_to_ply = '/Users/annikal/Documents/master_thesis/ProcessingLiDARdata/_out_Town03_190207_18/pc/173504.ply'
path_to_csv = '/Users/annikal/Documents/master_thesis/ProcessingLiDARdata/_out_Town03_190207_18/Town03_190207_18.csv'

# Load data:
pc, global_lidar_coordinates = load_data(path_to_ply, path_to_csv)
print('Shape of raw pointcloud: ', np.shape(pc))
print('global lidar coordinates: ', global_lidar_coordinates)

# PLOT RAW POINT CLOUD:
'''x = pc[:, 0]
y = pc[:, 1]
plt.plot(x, y, 'r.')
plt.axis('equal')
plt.title('Raw point cloud')
plt.show()'''

# Test array:
#pc = np.array([[-10,-10, 0], [-15, 3, 3], [26, 2, 3], [1, 2, 3], [10, 28, 9], [4, 1, -4], [4, 1, 5], [11, 9, 1], [12, 1, 10],[4, 8, 10], [1, 2, 3], [5, 7, 9], [25, 9, 1], [23, 3, 10]])
#print('Shape of test array: ', np.shape(pc))
#print(pc[0,:])
#pc = np.array([[-1,-1, 0], [1, 1, 0]])
#x = pc[:, 0]
#y = pc[:, 1]
#plt.plot(x, y, 'ro')
#plt.title('Raw point cloud')
#plt.show()
#print('Shape of raw pointcloud: ', np.shape(pc))


trimmed_pc = trim_pointcloud(pc)
print('Shape of trimmed pointcloud: ', np.shape(trimmed_pc))
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
plt.show()



'''rotated_pc = rotate_pointcloud_to_global(trimmed_pc, global_lidar_coordinates)
x_rot = rotated_pc[:, 0]
y_rot = rotated_pc[:, 1]
plt.plot(x_rot, y_rot, 'r.', x, y, 'b.')
plt.axis('equal')
plt.title('raw: blue, rotated: red')
plt.show()


translated_pc = translate_pointcloud_to_global(rotated_pc, global_lidar_coordinates)
x_trans = translated_pc[:, 0]
y_trans = translated_pc[:, 1]
plt.plot(x_trans, y_trans, 'r.', x, y, 'b.')
plt.axis('equal')
plt.title('translated: red, raw: blue')
plt.show()

print(global_lidar_coordinates)'''