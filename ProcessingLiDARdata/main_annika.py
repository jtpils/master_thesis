import numpy as np
from lidar_data_functions import *
from matplotlib import pyplot as plt
import pandas as pd
import random


path_to_lidar_data = '/home/master04/Desktop/Ply_files/_out_Town03_190306_1'
number_of_files_to_load_list = 1
dir_list = os.listdir(path_to_lidar_data)  # this should return a list where only one object is our csv_file
path_to_pc = os.path.join(path_to_lidar_data,
                          'pc/')  # we assume we follow the structure of creating lidar data folder with a pc folder for ply
for file in dir_list:
    if '.csv' in file:  # find csv-file
        path_to_csv = os.path.join(path_to_lidar_data, file)

translation = float(3)
rotation = float(15)
number_of_files_to_load = number_of_files_to_load_list

# create a list of all ply-files in a directory
ply_files = os.listdir(path_to_pc)

random.shuffle(ply_files)
for file_name in ply_files[:number_of_files_to_load]:
    print(file_name)
    file_name = '043471.ply'
    # Load data:
    try:
        # path_to_ply = path_to_pc + file_name
        path_to_ply = os.path.join(path_to_pc, file_name)
        pc, global_lidar_coordinates = load_data(path_to_ply, path_to_csv)
        #print('Creating sample ', i, ' of ', number_of_files_to_load)
    except:
        #print('Failed to load file ', file_name, '. Moving on to next file.')
        continue

    # create the sweep, transform a bit to create training sample
    rand_trans = random_rigid_transformation(translation, rotation)
    rand_trans = [2, -1, 0]
    print(rand_trans)
    sweep = training_sample_rotation_translation(pc, rand_trans)
    sweep = trim_pointcloud(sweep)
    # discretize and pad sweep

    # fake a map cutout
    cutout = trim_pointcloud(pc)

    plt.plot(cutout[0::2,0], cutout[0::2,1],'k.')
    plt.plot(sweep[0::2,0], sweep[0::2,1], 'b.')
    plt.plot([0, 2], [0, -1], 'r')
    plt.show()
