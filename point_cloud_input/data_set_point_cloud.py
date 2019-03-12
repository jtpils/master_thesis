import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from lidar_processing_functions import *

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


def get_grid(x, y, edges):
    k = 0
    for edge in edges[0]: # loop trough the x edges
        if x > edge:
            x_grid = k
        k = k + 1

    k = 0
    for edge in edges[0]:  # loop through the y edges
        if y > edge:
            y_grid = k
        k = k + 1

    return x_grid, y_grid


def get_neighbouring_grids(x_grid, y_grid):
    x_grids = [x_grid-1, x_grid, x_grid+1]
    y_grids = [y_grid-1, y_grid, y_grid+1]
    grid_name_list = list()
    for x in x_grids:
        for y in y_grids:
            grid_name = 'grid_' + str(x) + '_' + str(y)
            grid_name_list.append(grid_name)

    return grid_name_list


class PointCloudDataSet(Dataset):
    """Lidar sample dataset."""

    def __init__(self, data_set_path, number_of_samples):
        """
        Args:
            data_set_path (string): Path to directory with all the folder containing ply-files for each grid over town.
            number_of_samples (int): number of samples to create
        """

        self.data_set_path = data_set_path


        # maybe we should do all the hard work here and generate a list where we can check where our coordinates should be
        edges = np.load(os.path.join(data_set_path, 'edges.npy'))
        self.edges = edges
        del edges


        global_coordinates_path = os.path.join(data_set_path, 'global_coordinates.csv')
        self.global_coordinates_path = global_coordinates_path
        global_coordinates = pd.read_csv(global_coordinates_path)


        # generate list of which ply-files we will use in this data set. Chosen randomly from all available files.
        array_of_frame_numbers = global_coordinates['frame_number'].values  # which files exist
        number_of_available_ply_files = len(array_of_frame_numbers)  # how many exist

        # get random indices so that we can select our files
        selection_rule = np.random.choice(np.arange(number_of_available_ply_files), number_of_samples)  # consider what happens if we want more samples than there are ply-files
        array_of_frame_numbers = array_of_frame_numbers[selection_rule]
        list_of_ply_file_names =  get_file_name_from_frame_number(array_of_frame_numbers)
        self.list_of_sweeps_to_load = list_of_ply_file_names

        # keep a copy of the global coordinates that are JUST for the training sweeps
        self.sweeps_global_coordinates = global_coordinates[selection_rule]


        # create list with array of labels
        self.labels = [random_rigid_transformation(1, 2.5) for x in np.arange(number_of_samples)]

        del array_of_frame_numbers, number_of_available_ply_files, selection_rule, list_of_ply_file_names, global_coordinates


        del global_coordinates_path

    def __len__(self):
        return len(self.list_of_sweeps_to_load)

    def __getitem__(self, idx):

        # get the sweep we are looking for
        ply_file_name = self.list_of_sweeps_to_load[idx]
        ply_coordinates = self.sweeps_global_coordinates.iloc[idx].values  # get global coordinates for the sweep at row idx

        # find in which grid this ply-file exist
        x_grid, y_grid = get_grid(ply_coordinates[0], ply_coordinates[1], self.edges)  # global x, y
        grid_directory = self.data_set_path + '/grid_' + str(x_grid) + '_' + str(y_grid)

        # load the specific ply-file as our sweep.
        sweep, sweep_coordinates = load_data(os.path.join(grid_directory, ply_file_name), self.global_coordinates_path)
        # transform with global yaw
        sweep = rotate_pointcloud_to_global(sweep, ply_coordinates)
        # transform with random yaw
        sweep = rotate_point_cloud(sweep, self.labels[idx][2])


        # load all the files in range of our sweep.
        pc_super_array = np.zeros((1, 3))
        # get the neighbouring grids
        grids_to_load = get_neighbouring_grids(x_grid, y_grid)
        for grid in grids_to_load:
            grid_path = os.path.join(self.data_set_path, grid)
            try: # we might be trying to look into grid directories that do not exits, like grid_-1_2, or grid_1000_1
                list_ply_files = os.listdir(grid_path)
            except:
                continue

            for ply_file in list_ply_files:
                pc, global_coordinates = load_data(ply_file, self.global_coordinates_path)
                pc = rotate_pointcloud_to_global(pc, global_coordinates)
                pc_super_array = np.concatenate((pc_super_array, pc))

        # our initial guess in the map
        initial_guess = np.array((ply_coordinates[0]+self.labels[idx][0], ply_coordinates[1]+self.labels[idx][1]))
        # translate all the points in the super_array such that the initial guess becomes the origin


        # get a label


        training_sample = {'sweep': sweep, 'map': map, 'labels': self.labels[idx]}
        type(training_sample)

        return training_sample


def get_train_loader(batch_size, data_set_path, number_of_samples, kwargs):
    '''
    Get a training data loader.
    :param batch_size: batch size when making a forward pass through network
    :param data_set_path: Path to directory with all the folder containing ply-files for each grid over town.
    :param csv_path: Path to csv-file that describes the folder structure for the grids over town.
    :param number_of_samples: Number of samples to create in the dataset
    :param kwargs: use cpu or gpu
    :return: train_loader: data loader
    '''
    training_data_set = PointCloudDataSet(data_set_path, number_of_samples)
    n_training_samples = len(training_data_set)
    print('Number of training samples: ', n_training_samples)
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(training_data_set, batch_size=batch_size, sampler=train_sampler, num_workers=4, **kwargs)

    return train_loader


#def get_val_loader(data_set_path, csv_path, number_of_samples):


#def get_test_loader(data_set_path, csv_path, number_of_samples):
