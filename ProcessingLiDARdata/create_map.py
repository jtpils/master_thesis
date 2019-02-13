import numpy as np
from lidar_data_functions import *
from map_functions import *
from PIL import Image
import matplotlib.pyplot as plt

path_to_csv = '/home/master04/Desktop/_out/_out_Town03_190207_18/Town03_190207_18.csv'
path_to_ply_folder = '/home/master04/Desktop/_out/_out_Town03_190207_18/pc/'

# get list of all ply files in the ply_folder.
files_in_ply_folder = os.listdir(path_to_ply_folder)

pc_dict = {}
max_x_val = float("-inf")
max_y_val = float("-inf")
min_x_val = float("inf")
min_y_val = float("inf")

i = 0

channel_matrix = np.zeros([4, 600, 600])
for file in files_in_ply_folder[:1]:
    i = i+1
    path_to_ply = path_to_ply_folder + file
    point_cloud, global_coordinates = load_data(path_to_ply, path_to_csv)

    # rotate, translate the point cloud to global coordinates and trim the point cloud
    trimmed_pc = trim_pointcloud(point_cloud, range=50, roof=10, floor=-3)
    rotated_pc = rotate_pointcloud_to_global(trimmed_pc, global_coordinates)
    rotated_and_translated_pc = translate_pointcloud_to_global(rotated_pc, global_coordinates)

    # save the trimmed pc into a dictionary
    pc_dict["pc{0}".format(i)] = rotated_and_translated_pc

    # find max and min value of the x and y coordinates
    max_x_val_tmp = np.max(rotated_and_translated_pc[:, 0])
    min_x_val_tmp = np.min(rotated_and_translated_pc[:, 0])

    max_y_val_tmp = np.max(rotated_and_translated_pc[:, 1])
    min_y_val_tmp = np.min(rotated_and_translated_pc[:, 1])

    if max_x_val_tmp > max_x_val:
        max_x_val = max_x_val_tmp
        # print('max x', max_x_val)

    if min_x_val_tmp < min_x_val:
        min_x_val = min_x_val_tmp
        # print('min x', min_x_val)

    if max_y_val_tmp > max_y_val:
        max_y_val = max_y_val_tmp
        # print('max y', max_y_val)

    if min_y_val_tmp < min_y_val:
        min_y_val = min_y_val_tmp
        # print('min y', min_y_val)

    # discretize the cells in the map

    # channel_matrix = channel_matrix + discretize_pointcloud(rotated_and_translated_pc, spatial_resolution=0.05)

# array_to_png(channel_matrix)
# print(' max x', max_x_val, '\n', 'max y', max_y_val, '\n', 'min x', min_x_val, '\n', 'min y', min_y_val )

