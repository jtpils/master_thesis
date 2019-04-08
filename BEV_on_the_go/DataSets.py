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
        sweep = rotate_point_cloud(sweep, rand_trans[-1])
        sweep = translate_point_cloud(sweep, rand_trans[:2])
        sweep = trim_point_cloud_range(sweep, trim_range=15)
        sweep_image = discretize_point_cloud(sweep, trim_range=15, spatial_resolution=0.1, image_size=300)

        # create and normalize sample
        sample = np.concatenate((sweep_image, cutout_image))
        sample = normalize_sample(sample)
        training_sample = {'sample': torch.from_numpy(sample).float(),
                           'label': rand_trans}

        return training_sample


def get_loaders(path_training, path_training_csv, path_validation, path_validation_csv, batch_size, use_cuda):
    kwargs = {'pin_memory': True, 'num_workers': 16} if use_cuda else {'num_workers': 4}

    train_set = DataSetFakeData(path_training, path_training_csv)
    n_training_samples = 400#len(train_set)
    print('Number of training samples: ', n_training_samples)
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, **kwargs)

    val_set = DataSetFakeData(path_validation, path_validation_csv)
    n_val_samples = 20#len(val_set)
    print('Number of training samples: ', n_val_samples)
    val_sampler = SubsetRandomSampler(np.arange(n_val_samples, dtype=np.int64))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, **kwargs)

    return train_loader, val_loader
