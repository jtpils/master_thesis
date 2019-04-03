import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from preprocessing_data_functions import create_pillars , get_feature_tensor


class LiDARDataSet_PC(Dataset):
    """Lidar sample dataset."""

    def __init__(self, sample_dir):
        """
        Args:
            sample_dir <string>: Directory with all the samples.
            number_of_samples <int>: number of samples in the folder
        """
        self.sample_dir = sample_dir
        self.length = int((len(os.listdir(sample_dir))-1)/2)  # -1 for labels file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        path_sweep = 'sweep' + str(idx) + '.csv'
        path_sweep = os.path.join(self.sample_dir, path_sweep)
        sweep = pd.read_csv(path_sweep).values
        sweep_pillars, sweep_coordinates = create_pillars(sweep)
        sweep_features , sweep_coordinates = get_feature_tensor(sweep_pillars, sweep_coordinates)

        path_cutout = 'cutout' + str(idx) + '.csv'
        path_cutout =  os.path.join(self.sample_dir, path_cutout)
        cutout = pd.read_csv(path_cutout).values
        cutout_pillars, cutout_coordinates = create_pillars(cutout)
        cutout_features, cutout_coordinates = get_feature_tensor(cutout_pillars, cutout_coordinates)

        path_labels = os.path.join(self.sample_dir, 'labels.csv')
        labels = pd.read_csv(path_labels).values

        label = labels[idx, :]

        training_sample = {'sweep': sweep_features,'sweep_coordinates': sweep_coordinates, 'cutout': cutout_features,
                           'cutout_coordinates': cutout_coordinates, 'labels': label}

        return training_sample


def get_train_loader_pc(batch_size, data_set_path, kwargs):
    '''
    Create training data loader.
    :param batch_size: batch size when making a forward pass through network
    :param data_set_path: Path to directory with all the folder containing ply-files for each grid over town.
    :param kwargs: use cpu or gpu
    :return: train_loader: data loader
    '''

    training_data_set = LiDARDataSet_PC(data_set_path)
    print('Number of training samples: ', len(training_data_set))
    train_sampler = SubsetRandomSampler(np.arange(len(training_data_set), dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(training_data_set, batch_size=batch_size, sampler=train_sampler, **kwargs)

    return train_loader
