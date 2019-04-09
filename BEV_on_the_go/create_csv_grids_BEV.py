import numpy as np
import csv
import pandas as pd
import os
import shutil
from tqdm import tqdm
from functions import *


# This script creates one csv-file per grid, with all the detections in the grid stored in csv-file. The detections are
# saved in global coordinates.

save_data_path = '/home/master04/Desktop/Dataset/ply_grids/csv_grids_190409/csv_grids_training/' # save new data to this path
path = '/home/master04/Desktop/Dataset/ply_grids/training_sorted_grid_ply/' # path to your ply-grids

edges = np.load(path + 'edges.npy')
shutil.copyfile(path+'edges.npy', save_data_path+'edges.npy')
path_global_csv = path+'global_coordinates.csv'
#path_global_csv = '/home/master04/Desktop/Dataset/ply_grids/in_global_coords/validation/validation_set.csv'
#shutil.copyfile(path+'global_coordinates.csv', path_global_csv)
shutil.copyfile(path_global_csv, save_data_path+'global_coordinates.csv')

# go through all directories, and for each directory, create a super-csv with all the detections from that directory (translated and rotated to global coordinates.)
directory_path_list = [f.path for f in os.scandir(path) if f.is_dir()]
directory_name_list = [f.name for f in os.scandir(path) if f.is_dir()]
i = 0
for i in tqdm(np.arange(len(directory_path_list))):
    file_list = os.listdir(directory_path_list[i])
    if len(file_list) is not 0: #if there are files, do some stuff!
        # Create a csv file
        csv_name = directory_name_list[i] + '.csv'
        csv_path = os.path.join(save_data_path, csv_name)
        with open(csv_path , mode = 'w') as csv_file:
            fieldnames = ['x', 'y', 'z']
            csv_writer = csv.writer(csv_file, delimiter = ',' , quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(fieldnames)

        pc_super_array = np.zeros((1, 3))  # zeros are removed later
        for file in file_list:
            path_to_ply = os.path.join(directory_path_list[i], file)
            pc, global_lidar_coordinates = load_data(path_to_ply, path_global_csv)
            pc = trim_point_cloud_range(pc, origin=global_lidar_coordinates[:2], trim_range=30)
            pc = trim_point_cloud_vehicle_ground(pc, origin=global_lidar_coordinates[:2], remove_vehicle=True, remove_ground=False)
            pc_super_array = np.concatenate((pc_super_array, pc))
        pc_super_array = pc_super_array[1:,:]  # remove the first row of zeros

        with open(csv_path , mode = 'a') as csv_file_2:
            csv_writer_2 = csv.writer(csv_file_2, delimiter = ',' , quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in np.arange(len(pc_super_array)):
                x = pc_super_array[row,0]
                y = pc_super_array[row,1]
                z = pc_super_array[row,2]
                csv_writer_2.writerow([x, y, z])


print(' ')
