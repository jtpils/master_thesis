import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import time
from torch.utils.data.sampler import SubsetRandomSampler

import pickle


class LiDARDataSet_PC(Dataset):
    """Lidar sample dataset, load from existing sample files."""

    def __init__(self, sample_dir, number_of_samples):
        """
        Args:
            sample_dir (string): Directory with all the samples.
        """
        self.sample_dir = sample_dir
        self.length = len(os.listdir(sample_dir))
        self.number_of_samples = number_of_samples

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        file_name = 'training_sample_' + str(idx)

        sample_file = os.path.join(self.sample_dir, file_name)
        pickle_in = open(sample_file, "rb")
        training_sample = pickle.load(pickle_in)

        return training_sample


def get_train_loader_pc(batch_size, data_set_path, number_of_samples, kwargs):
    '''
    Get a training data loader.
    :param batch_size: batch size when making a forward pass through network
    :param data_set_path: Path to directory with all the folder containing ply-files for each grid over town.
    :param csv_path: Path to csv-file that describes the folder structure for the grids over town.
    :param number_of_samples: Number of samples to create in the dataset
    :param kwargs: use cpu or gpu
    :return: train_loader: data loader
    '''

    training_data_set = LiDARDataSet_PC(data_set_path, number_of_samples)
    n_training_samples = number_of_samples
    print('Number of training samples: ', n_training_samples)
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(training_data_set, batch_size=batch_size, sampler=train_sampler, **kwargs)

    return train_loader
