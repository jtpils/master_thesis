import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import os
from functions import *


class DataSetFakeData(Dataset):
    """ Fake data, sweep+sweep """

    def __init__(self, path_pc, csv_path, translation=1, rotation=0):

        self.path_pc = path_pc
        self.ply_files = os.listdir(self.path_pc)
        self.path_csv = csv_path
        self.translation = translation
        self.rotation = rotation

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx):
        # get point cloud
        ply_path = os.path.join(self.path_pc, self.ply_files[idx])
        pc, global_coords = load_data(ply_path, self.path_csv)

        # create fake cutout with no translation or rotation
        cutout = trim_point_cloud_range(pc, trim_range=15)
        cutout = trim_point_cloud_vehicle_ground(cutout, remove_vehicle=True, remove_ground=False)
        cutout_image = discretize_point_cloud(cutout, trim_range=15, spatial_resolution=0.1, image_size=300)

        # rotate and translate sweep
        rand_trans = random_rigid_transformation(self.translation, self.rotation)
        sweep = trim_point_cloud_range(pc, trim_range=20)
        sweep = trim_point_cloud_vehicle_ground(sweep, remove_vehicle=True, remove_ground=False)
        sweep = rotate_point_cloud(sweep, rand_trans[-1], to_global=False)
        sweep = translate_point_cloud(sweep, rand_trans[:2])
        sweep = trim_point_cloud_range(sweep, trim_range=15)
        sweep_image = discretize_point_cloud(sweep, trim_range=15, spatial_resolution=0.1, image_size=300)

        # create and normalize sample
        sample = np.concatenate((sweep_image, cutout_image))
        sample = normalize_sample(sample)
        training_sample = {'sample': torch.from_numpy(sample).float(),
                           'label': rand_trans}

        return training_sample


class DataSetMapData(Dataset):
    """Generate "real" data on the go, eg sweep+map-cut-out, using the map.npy file"""

    def __init__(self, sample_path, csv_path, map_path, minmax_path, translation=1, rotation=0):
        """
        Args:
            sample_path (string): Directory with all the sweeps.
            csv_path to csv with global coordinates
            map_path: path to map.npy
            minmax_path: path to min max values in the map
            translation, rotation; the amount of rigid transformation
        """
        self.sample_dir = sample_path
        self.sweeps_file_names = os.listdir(sample_path)
        self.csv_path = csv_path
        self.map = np.load(map_path)
        self.map_minmax = np.load(minmax_path)
        self.translation = translation
        self.rotation = rotation
        print('Done initializing data set.')

    def __len__(self):
        return len(self.sweeps_file_names)

    def __getitem__(self, idx):
        # Load one sweep and generate a random transformation
        file_name = self.sweeps_file_names[idx]
        pc, global_coords = load_data(os.path.join(self.sample_dir,file_name), self.csv_path)
        rand_trans = random_rigid_transformation(self.translation, self.rotation)

        # sweep
        # rotate and translate sweep
        rand_trans = random_rigid_transformation(self.translation, self.rotation)
        sweep = trim_point_cloud_range(pc, trim_range=20)
        sweep = trim_point_cloud_vehicle_ground(sweep, remove_vehicle=True, remove_ground=False)
        sweep = rotate_point_cloud(sweep, rand_trans[-1], to_global=False)
        sweep = translate_point_cloud(sweep, rand_trans[:2])
        sweep = trim_point_cloud_range(sweep, trim_range=15)
        sweep_image = discretize_point_cloud(sweep, trim_range=15, spatial_resolution=0.1, image_size=300)

        # map cut-out
        cut_out_coordinates = global_coords[:2]
        spatial_resolution = 0.1
        x_min, x_max, y_min, y_max = self.map_minmax
        x_grid = np.floor((cut_out_coordinates[0]-x_min)/spatial_resolution).astype(int)
        y_grid = np.floor((cut_out_coordinates[1]-y_min)/spatial_resolution).astype(int)
        cutout_image = self.map[:, x_grid-150:x_grid+150, y_grid-150:y_grid+150]

        # concatenate the sweep and the cutout image into one image and save.
        sweep_and_cutout_image = np.concatenate((sweep_image, cutout_image))
        sweep_and_cutout_image = normalize_sample(sweep_and_cutout_image)

        training_sample = {'sample': torch.from_numpy(sweep_and_cutout_image).float(), 'label': rand_trans}
        return training_sample


def get_loaders(path_training, path_training_csv, path_validation, path_validation_csv, batch_size, use_cuda):
    kwargs = {'pin_memory': True, 'num_workers': 16} if use_cuda else {'num_workers': 0}

    # USE MAP-CUTOUTS
    '''
    map_train_path = '/home/annika_lundqvist144/maps/map_Town01/map.npy'
    map_minmax_train_path = '/home/annika_lundqvist144/maps/map_Town01/max_min.npy'
    '''
    map_train_path = '/home/master04/Desktop/Maps/map_Town01/map.npy'
    map_minmax_train_path = '/home/master04/Desktop/Maps/map_Town01/max_min.npy'
    train_set = DataSetMapData(path_training, path_training_csv, map_train_path, map_minmax_train_path)

    # USE FAKA DATA
    #train_set = DataSetFakeData(path_training, path_training_csv)

    n_training_samples = 10#len(train_set)
    print('Number of training samples: ', n_training_samples)
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, **kwargs)

    # USE MAP-CUTOUTS
    '''
    map_val_path = '/home/annika_lundqvist144/maps/map_Town02_validation/map.npy'
    map_minmax_val_path = '/home/annika_lundqvist144/maps/map_Town02_validation/max_min.npy'
    '''
    map_val_path = '/home/master04/Desktop/Maps/map_Town02_validation/map.npy'
    map_minmax_val_path = '//home/master04/Desktop/Maps/map_Town02_validation/max_min.npy'
    val_set = DataSetMapData(path_validation, path_validation_csv, map_val_path, map_minmax_val_path)

    # USE FAKA DATA
    #val_set = DataSetFakeData(path_validation, path_validation_csv)

    n_val_samples = 10#len(val_set)
    print('Number of validation samples: ', n_val_samples)
    val_sampler = SubsetRandomSampler(np.arange(n_val_samples, dtype=np.int64))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, **kwargs)

    return train_loader, val_loader
