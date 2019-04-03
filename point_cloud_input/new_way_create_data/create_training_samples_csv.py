import os
import numpy as np
import pandas as pd
from lidar_processing_functions import *
import csv


# This script creates training samples: one csv-file with sweep, and one csv-file with all detections in that area.

def get_neighbouring_grids(x_grid, y_grid):
    # some grids will be outside the scope, as in negative or too large, but we will handle that with try/except in the script
    x_grids = [x_grid-1, x_grid, x_grid+1]
    y_grids = [y_grid-1, y_grid, y_grid+1]
    grid_name_list = list()
    for x in x_grids:
        for y in y_grids:
            grid_name = 'grid_' + str(x) + '_' + str(y)
            grid_name_list.append(grid_name)

    return grid_name_list


def get_file_name_from_frame_number(frame_number_array):
    '''
    return a file name as a string given the file's frame number.
    :param frame_number: array with ints
    :return: list with strings
    '''
    file_name_list = list()
    # handle that input is most likely an array
    for frame_number in frame_number_array:
        # count number of digits in frame_number. Zero pad at beginning until we have 6 chars. Append '.ply'
        frame_number_string = str(frame_number)
        number_of_char = len(frame_number_string)
        add_number_of_zeros = 6 - number_of_char
        file_name = '0'*add_number_of_zeros + frame_number_string + '.ply'
        file_name_list.append(file_name)

    return file_name_list

# path to all ply-grid-files, we will pick our sweeps from here
ply_path = '/home/master04/Desktop/Dataset/ply_grids/Town01_sorted_grid_ply'

# path to corresponding csv-grid-files
data_set_path = '/home/master04/Desktop/Dataset/ply_grids/in_global_coords/Town01/'


save_data_path = '/home/master04/Desktop/Dataset/point_cloud/new_set/'
number_of_samples = 2
grid_size = 15

# load all global coordinates for all ply-files
edges = np.load(os.path.join(data_set_path, 'edges.npy'))
global_coordinates_path = os.path.join(data_set_path, 'global_coordinates.csv')
global_coordinates = pd.read_csv(global_coordinates_path)

# generate a list of which ply-files we will use in this data set. Chosen randomly from all available files.
array_of_frame_numbers = global_coordinates['frame_number'].values  # which files exist
# get indices so that we can select our files
selection_rule = np.random.choice(np.arange(len(array_of_frame_numbers)), number_of_samples)  # consider what happens if we want more samples than there are ply-files
array_of_frame_numbers = array_of_frame_numbers[selection_rule]
list_of_sweeps_to_load = get_file_name_from_frame_number(array_of_frame_numbers)

# keep a copy of the global coordinates that are JUST for the training sweeps
sweeps_global_coordinates = global_coordinates.values[selection_rule, :]

# create list with array of labels
labels = [random_rigid_transformation(1, 0) for x in np.arange(number_of_samples)]
#### SAVE LABELS AS CSV FILE!!!!!!! #####
csv_name = 'labels.csv'
csv_path = os.path.join(save_data_path, csv_name)
with open(csv_path , mode = 'w') as csv_file:
    fieldnames = ['x', 'y', 'z']
    csv_writer = csv.writer(csv_file, delimiter = ',' , quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(fieldnames)

with open(csv_path , mode = 'a') as csv_file_2:
    csv_writer_2 = csv.writer(csv_file_2, delimiter = ',' , quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in np.arange(len(labels)):
        x = labels[row][0]
        y = labels[row][1]
        z = labels[row][2]
        csv_writer_2.writerow([x, y, z])







#for ply_file_name in list_of_sweeps_to_load:
for idx in np.arange(len(list_of_sweeps_to_load)):
    ply_file_name = list_of_sweeps_to_load[idx]
    ply_coordinates = sweeps_global_coordinates[idx, :]  # get global coordinates for the sweep at row idx

    # find in which grid this ply-file exist
    x_grid = int(np.floor((ply_coordinates[1]-edges[0][0])/grid_size))
    y_grid = int(np.floor((ply_coordinates[2]-edges[1][0])/grid_size))
    grid_directory = ply_path + '/grid_' + str(x_grid) + '_' + str(y_grid)

    # load the specific ply-file as our sweep.
    sweep, sweep_coordinates = load_data(os.path.join(grid_directory, ply_file_name), global_coordinates_path)
    # WE SHOULD DO THE TWO ROTATIONS AT THE SAME TIME, eg yaw+rand
    # transform with global yaw
    sweep = rotate_pointcloud_to_global(sweep, sweep_coordinates)
    # transform with random yaw
    sweep = rotate_point_cloud(sweep, labels[idx][2])
    sweep = trim_pointcloud(sweep)

    #sample points so that all training samples are of the same size always
    max_num_points_sweep = 3000
    if len(sweep) > max_num_points_sweep:
        selection_rule = np.random.choice(np.arange(len(sweep)), max_num_points_sweep)
        sweep = sweep[selection_rule, :]
    else:
        zeros = np.zeros((max_num_points_sweep-len(sweep), 3))
        sweep = np.concatenate((sweep, zeros), 0)

    csv_name = 'sweep' + str(idx) + '.csv'
    csv_path = os.path.join(save_data_path, csv_name)
    with open(csv_path , mode = 'w') as csv_file:
        fieldnames = ['x', 'y', 'z']
        csv_writer = csv.writer(csv_file, delimiter = ',' , quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(fieldnames)

    with open(csv_path , mode = 'a') as csv_file_2:
        csv_writer_2 = csv.writer(csv_file_2, delimiter = ',' , quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in np.arange(len(sweep)):
            x = sweep[row,0]
            y = sweep[row,1]
            z = sweep[row,2]
            csv_writer_2.writerow([x, y, z])

    # get the map-cut-out too
    pc_super_array = np.zeros((1, 3))  # zeros are removed later
    # get the neighbouring grids
    grids_to_load = get_neighbouring_grids(x_grid, y_grid)
    for grid in grids_to_load:
        grid_path = data_set_path + grid + '.csv'
        try:  # we might be trying to look into grid csvs that do not exits, like grid_-1_2, or grid_1000_1, or empty grids
            pc = pd.read_csv(grid_path)
        except:
            continue
        pc_super_array = np.concatenate((pc_super_array, pc))
    pc_super_array = pc_super_array[1:,:]  # remove the first row of zeros
    # our initial guess in the map
    initial_guess = np.array((ply_coordinates[1]+labels[idx][0], -(ply_coordinates[2]+labels[idx][1]), 0))
    # translate all the points in the super_array such that the initial guess becomes the origin
    pc_super_array = pc_super_array - initial_guess
    map_cutout = trim_pointcloud(pc_super_array)

    max_num_points_map = 30000
    if len(map_cutout) > max_num_points_map:
        selection_rule = np.random.choice(np.arange(len(map_cutout)), max_num_points_map)
        map_cutout = map_cutout[selection_rule, :]
    else:
        zeros = np.zeros((max_num_points_map-len(map_cutout), 3))
        map_cutout = np.concatenate((map_cutout, zeros), 0)

    csv_name = 'cutout' + str(idx) + '.csv'
    csv_path = os.path.join(save_data_path, csv_name)
    with open(csv_path , mode = 'w') as csv_file:
        fieldnames = ['x', 'y', 'z']
        csv_writer = csv.writer(csv_file, delimiter = ',' , quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(fieldnames)

    with open(csv_path , mode = 'a') as csv_file_2:
        csv_writer_2 = csv.writer(csv_file_2, delimiter = ',' , quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in np.arange(len(map_cutout)):
            x = map_cutout[row,0]
            y = map_cutout[row,1]
            z = map_cutout[row,2]
            csv_writer_2.writerow([x, y, z])




# How to read samples
#sweep  = pd.read_csv('/home/master04/Desktop/Dataset/point_cloud/new_set/sweep0.csv', delimiter=',').values
