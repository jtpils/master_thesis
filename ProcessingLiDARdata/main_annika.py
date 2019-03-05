import numpy as np
from lidar_data_functions import *
from matplotlib import pyplot as plt
import pandas as pd


#path_to_ply = '/home/master04/Desktop/_out_town2/pc/059176.ply'
#path_to_csv = '/home/master04/Desktop/_out_town2/town2.csv'

################################# CHANGE HERE ###########################################################
'''print(' ')
path_to_lidar_data = '/Users/annikal/Documents/master_thesis/ProcessingLiDARdata/_out_Town02_190221_1'
dir_list = os.listdir(path_to_lidar_data)
path_to_pc = os.path.join(path_to_lidar_data, 'pc/')  # we assume we follow the structure of creating lidar data folder with a pc folder for ply
for file in dir_list:
    if '.csv' in file:  # find csv-file
        path_to_csv = os.path.join(path_to_lidar_data, file)


# create a list of all ply-files in a directory
ply_files = os.listdir(path_to_pc)

i = 0

number_of_files_to_load = 1
for file_name in ply_files[:number_of_files_to_load]:

    # Load data:
    try:
        path_to_ply = path_to_pc + file_name
        pc, global_lidar_coordinates = load_data(path_to_ply, path_to_csv)
        i = i + 1
        print('Creating training sample ', i, ' of ', number_of_files_to_load)
        print(file_name)
    except:
        print('Failed to load file ', file_name, '. Moving on to next file.')
        continue


    sweep = trim_pointcloud(pc)
    sweep = discretize_pointcloud(sweep, array_size=600, trim_range=15, spatial_resolution=1, padding=False, pad_size=150)
    sweep_and_sweep = np.concatenate((sweep, sweep))
    sweep_and_sweep = normalize_sample(sweep_and_sweep)
'''


test_array = np.zeros((8,3,3))
for i in np.arange(8):
    test_array[i, :, :] = np.ones((3,3)) * i

print(np.shape(test_array))
test_array = normalize_sample(test_array)

