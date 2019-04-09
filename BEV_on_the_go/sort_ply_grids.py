import os
import pandas as pd
import numpy as np
import shutil
import csv
from tqdm import tqdm
from functions import *


def get_file_name_from_frame_number(frame_number):
    frame_number_string = str(int(frame_number))
    number_of_char = len(frame_number_string)
    add_number_of_zeros = 6 - number_of_char
    file_name = '0'*add_number_of_zeros + frame_number_string + '.ply'

    return file_name


# **********# change the rows marked with this # **********#

path_to_ply = '/home/master04/Desktop/Ply_files/_out_Town01_190402_1'  # **********#
path_to_csv = os.path.join(path_to_ply, 'Town01_190402_1.csv')  # **********#
path_to_pc = os.path.join(path_to_ply, 'pc')

new_folder = '/home/master04/Desktop/Dataset/ply_grids/training_sorted_grid_ply'  # **********#
os.mkdir(new_folder)

files_in_ply_folder = os.listdir(path_to_pc)
pc_super_array = np.zeros((1, 3))
for file in tqdm(files_in_ply_folder):
    try:
        # Create the path to the ply file
        path_to_ply = os.path.join(path_to_pc, file)
        pc, global_coordinates = load_data(path_to_ply, path_to_csv)
    except:
        print('Failed to load file ', file, '. Moving on to next file.')
        continue

    pc = trim_point_cloud_range(pc, origin=global_coordinates[:2], trim_range=30)
    pc = trim_point_cloud_vehicle_ground(pc, origin=global_coordinates[:2], remove_vehicle=True, remove_ground=False)
    pc_super_array = np.concatenate((pc_super_array, pc))

print('Done loading files. Creating map.')
pc_super_array = np.delete(pc_super_array, 0, axis=0)
min_max = [np.min(pc_super_array[:,0]), np.max(pc_super_array[:,0]), np.min(pc_super_array[:,1]), np.max(pc_super_array[:,1])]
min_x, max_x, min_y, max_y = min_max


grid_size = 15
number_x_grids = int(np.ceil((max_x - min_x)/grid_size))
number_of_x_edges = number_x_grids + 1
x_edges = np.arange(number_of_x_edges)*grid_size+min_x

number_y_grids = int(np.ceil((max_y - min_y)/grid_size))
number_of_y_edges = number_y_grids + 1
y_edges = np.arange(number_of_y_edges)*grid_size+min_y

csv_edges_path = os.path.join(new_folder, 'edges.npy')
np.save(csv_edges_path, [x_edges, y_edges])

# create all subdirectories to store ply-files in
for x in np.arange(number_x_grids):
    for y in np.arange(number_y_grids):
        grid_name = new_folder + '/grid_' + str(x) + '_' + str(y)
        os.mkdir(grid_name)


# create csv, where we store all frame numbers and global coordinates
new_csv_global_coordinates_path = os.path.join(new_folder, 'global_coordinates.csv')
with open(new_csv_global_coordinates_path, mode='w') as csv_file:
    fieldnames = ['frame_number', 'x', 'y', 'z', 'yaw']
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(fieldnames)
    print('Successfully created global_coordinates.csv')

 # COPY ADN RENAME THE GLOBAL COORDINATES HERE INSTEAD
#source =
##destination = new_csv_global_coordinates_path
#shutil.copyfile(source, destination)


global_coordinates = pd.read_csv(path_to_csv)
global_coordinates['y'] = -global_coordinates['y']
# loop trough each ply file and copy it to the correct directory
for row in tqdm(global_coordinates.values):
    file_name = get_file_name_from_frame_number(row[0])

    try:  # because we are out of bounds sometimes??
        x_grid = int(np.floor((row[1]-min_x)/grid_size))
        y_grid = int(np.floor((row[2]-min_y)/grid_size))
        grid_name = new_folder + '/grid_' + str(x_grid) + '_' + str(y_grid)
        source = os.path.join(path_to_pc, file_name)
        destination = os.path.join(grid_name, file_name)

        try:  # beacuse some files seem to be missing
            shutil.copyfile(source, destination)

            # write coordinates from the old csv_file (already available in variable row)
            ''' #THIS YIELDS ONLY POSITIVE VALUES, since we change sign before this for-loop to get the grids rights.
             #COPY FILE FROM PLY-FOLDER INSTEAD.
            with open(new_csv_global_coordinates_path, mode='a') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([int(row[0]), row[1], row[2], row[3], row[4]])
            '''

        except:
            print('oops 1')
            continue

    except:
        print('oops 2')
        continue
