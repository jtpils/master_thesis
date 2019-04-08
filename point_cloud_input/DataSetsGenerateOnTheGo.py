from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from preprocessing_data_functions import create_pillars , get_feature_tensor
from lidar_processing_functions import *


class LiDARDataSet_PC_fake_data(Dataset):
    """Lidar sample dataset."""

    def __init__(self, sample_dir, csv_path, translation, rotation):
        """
        Args:
            sample_dir <string>: Directory with all ply-files.
        """
        self.sample_dir = sample_dir
        self.list_of_files = os.listdir(self.sample_dir)
        self.length = len(self.list_of_files)
        self.csv_path = csv_path
        self.labels = [random_rigid_transformation(translation, rotation) for x in np.arange(self.length)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # load ply-file
        file_path = os.path.join(self.sample_dir, self.list_of_files[idx])
        pc, global_coord = load_data(file_path, self.csv_path)
        label = self.labels[idx]

        # cut-out
        cutout = pc + np.array((label[0], label[1], 0))
        cutout = trim_pointcloud(cutout)
        cutout_pillars, cutout_coordinates = create_pillars(cutout)
        cutout_features, cutout_coordinates = get_feature_tensor(cutout_pillars, cutout_coordinates)

        # rotate/translate sweep, create pillars/tensor
        sweep = rotate_point_cloud(pc, label[-1])
        sweep = trim_pointcloud(sweep)
        sweep_pillars, sweep_coordinates = create_pillars(sweep)
        sweep_features , sweep_coordinates = get_feature_tensor(sweep_pillars, sweep_coordinates)

        # save labels, sweep + rot/trans-sweep
        training_sample = {'sweep': torch.from_numpy(sweep_features).float(),'sweep_coordinates': sweep_coordinates,
                           'cutout': torch.from_numpy(cutout_features).float(),
                           'cutout_coordinates': cutout_coordinates, 'labels': label}

        return training_sample


class LiDARDataSet_PC(Dataset):
    """Lidar sample dataset."""

    def __init__(self, sample_dir, csv_path, grid_csv_path, translation, rotation):
        """
        Args:
            sample_dir <string>: Directory with all ply-files.
        """
        self.sample_dir = sample_dir
        self.list_of_files = os.listdir(self.sample_dir)
        self.length = len(self.list_of_files)
        self.csv_path = csv_path
        self.labels = [random_rigid_transformation(translation, rotation) for x in np.arange(self.length)]


        list_of_csv = os.listdir(grid_csv_path)
        sweeps = []
        print('loading all LiDAR detections...')
        for file in tqdm(list_of_csv):
            if 'grid' in file:
                pc = pd.read_csv(os.path.join(grid_csv_path, file))
                sweeps.append(pc)
        self.lidar_points = pd.concat(sweeps)
        print('Done loading detections.')
        del sweeps, pc

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # load ply-file
        file_path = os.path.join(self.sample_dir, self.list_of_files[idx])
        pc, global_coord = load_data(file_path, self.csv_path)
        label = self.labels[idx]

        # map cut-out
        cut_out_coordinates = global_coord[0][:2] + label[:2]  # translation x, y
        # we want all coordinates that in trim_range around cut_out_coordinates
        trim_range = 15
        # get all points around the sweep
        cutout = self.lidar_points[self.lidar_points['x'] <= cut_out_coordinates[0]+trim_range]
        cutout = cutout[cutout['x'] >= cut_out_coordinates[0]-trim_range]
        cutout = cutout[cutout['y'] <= cut_out_coordinates[1]+trim_range]
        cutout = cutout[cutout['y'] >= cut_out_coordinates[1]-trim_range]
        # if we want to use occupancy grid, sample points first
        # move all points such that the cut-out-coordinates becomes the origin
        cutout = cutout.values - np.array((cut_out_coordinates[0], cut_out_coordinates[1], 0))
        cutout_pillars, cutout_coordinates = create_pillars(cutout)
        cutout_features, cutout_coordinates = get_feature_tensor(cutout_pillars, cutout_coordinates)

        # rotate/translate sweep, create pillars/tensor
        sweep = rotate_point_cloud(pc, label[-1])
        sweep = trim_pointcloud(sweep)
        sweep_pillars, sweep_coordinates = create_pillars(sweep)
        sweep_features , sweep_coordinates = get_feature_tensor(sweep_pillars, sweep_coordinates)

        # save labels, sweep + rot/trans-sweep
        training_sample = {'sweep': torch.from_numpy(sweep_features).float(),'sweep_coordinates': sweep_coordinates,
                           'cutout': torch.from_numpy(cutout_features).float(),
                           'cutout_coordinates': cutout_coordinates, 'labels': label}

        return training_sample


def get_train_loader_pointpillars(batch_size, data_set_path, csv_path, rotation, translation, kwargs):
    '''
    Create training data loader.
    :param batch_size: batch size when making a forward pass through network
    :param data_set_path: Path to directory with all the folder containing ply-files for each grid over town.
    :param kwargs: use cpu or gpu
    :return: train_loader: data loader
    '''
    grid_csv_path = '/home/master04/Desktop/Dataset/ply_grids/in_global_coords/Town01'
    training_data_set = LiDARDataSet_PC(data_set_path, csv_path, grid_csv_path, translation, rotation)
    print('Number of training samples: ', len(training_data_set))
    train_sampler = SubsetRandomSampler(np.arange(len(training_data_set), dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(training_data_set, batch_size=batch_size, sampler=train_sampler, drop_last = True, **kwargs)

    return train_loader
