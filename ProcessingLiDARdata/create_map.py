import numpy as np
from lidar_data_functions import *
from map_functions import *
from matplotlib import pyplot as plt

path_to_ply_folder = '/home/master04/Desktop/_out_Town02_190221_1/pc/'
path_to_csv = '/home/master04/Desktop/_out_Town02_190221_1/Town02_190221_1.csv'

#files_in_ply_folder = os.listdir(path_to_ply_folder)
# pc_dict = {}
files_in_ply_folder = ['003119.ply']
pc_super_array = np.zeros((1, 3))
print(np.shape(pc_super_array))
max_x_val = float("-inf")
max_y_val = float("-inf")
min_x_val = float("inf")
min_y_val = float("inf")
i = 0

for file in files_in_ply_folder:  # the last number is how large steps to take
    # i = i+1  # this parameter is used if storing each ply file in a dict

    # Create the path to the ply file
    path_to_ply = path_to_ply_folder + file
    point_cloud, global_coordinates = load_data(path_to_ply, path_to_csv)

    # rotate, translate the point cloud to global coordinates and trim the point cloud
    trimmed_pc = trim_pointcloud(point_cloud, range=500, roof=11, floor=0)
    rotated_pc = rotate_pointcloud_to_global(trimmed_pc, global_coordinates)
    rotated_and_translated_pc = translate_pointcloud_to_global(rotated_pc, global_coordinates)

    # concatenate the rotated and translated array into a super array
    pc_super_array = np.concatenate((pc_super_array, rotated_and_translated_pc))

    # Here one can save the trimmed pc into a dict
    # pc_dict["pc{0}".format(i)] = rotated_and_translated_pc

    # find max and min value of the x and y coordinates
    # If using the super array this can be removed and the max and min values can be found from the super_array outside
    # the for loop.
    max_x_val_tmp = np.max(rotated_and_translated_pc[:, 0])
    min_x_val_tmp = np.min(rotated_and_translated_pc[:, 0])

    max_y_val_tmp = np.max(rotated_and_translated_pc[:, 1])
    min_y_val_tmp = np.min(rotated_and_translated_pc[:, 1])

    if max_x_val_tmp > max_x_val:
        max_x_val = max_x_val_tmp

    if min_x_val_tmp < min_x_val:
        min_x_val = min_x_val_tmp

    if max_y_val_tmp > max_y_val:
        max_y_val = max_y_val_tmp

    if min_y_val_tmp < min_y_val:
        min_y_val = min_y_val_tmp


# delete the zeros in the first row in the super array that was used at the initalization of the array.
pc_super_array = np.delete(pc_super_array, 0, axis=0)

print(np.shape(pc_super_array))
print(pc_super_array[1:20,:2])

x = pc_super_array[:, 0]
y = pc_super_array[:, 1]
#z = pc_super_array[:, 2]

plt.plot(x , y, 'b.')
plt.ylabel('y')
plt.xlabel('x')
plt.show()



'''

# save the max and min values in an array. This is used to decide the size of the map
min_max = np.array((min_x_val, max_x_val, min_y_val, max_y_val))

# discretice the point cloud.
discretized_pc = discretize_pointcloud_map(pc_super_array, min_max)






# Save the discretized map in a folder.
#array_to_png(discretized_pc)

# show map
# NORMALIZE THE BEV IMAGE

# max_value = np.max(discretized_pc[1, :, :])
# print('Max max_value inarray_to_png: ', max_value)

# avoid division with 0
# if max_value == 0:
#     max_value = 1

# scale = 255/max_value
# discretized_pc[1, :, :] = discretized_pc[1, :, :] * scale
# print('Largest pixel value (should be 255) : ', np.max(discretized_pc[1, :, :]))

# img = Image.fromarray(discretized_pc[1, :, :])
# new_img = img.convert("L")
# new_img.show()


'''
