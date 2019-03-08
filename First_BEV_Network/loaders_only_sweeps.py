import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from cut_out_from_map import get_cut_out
from torch.utils.data.sampler import SubsetRandomSampler
from LiDARDataSet import LiDARDataSet


class LiDARDataSet_onthego(Dataset):
    """Lidar sweep dataset. The cutouts are generated on the go by the loaders"""

    def __init__(self, csv_file, sweep_dir, map_path, map_minmax_values_path):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            sweep_dir (string): Directory with all the samples.
            map_path (string): path to map.npy-file
            map_minmax_values_path (string): path to .npy file with minmax values of map
        """

        self.csv_labels = pd.read_csv(csv_file, usecols=['x', 'y', 'angle'])
        self.sweep_dir = sweep_dir
        self.cut_out_coordinates = pd.read_csv(csv_file, usecols=['x_map', 'y_map'])
        self.map = np.load(map_path)
        self.map_minmax_values = np.load(map_minmax_values_path)

    def __len__(self):
        return len(self.csv_labels)

    def __getitem__(self, idx):

        sweep_file = os.path.join(self.sweep_dir, str(idx))
        sweep = torch.from_numpy(np.load(sweep_file + '.npy')).float()

        labels = self.csv_labels.iloc[idx-1, :].values
        cut_out_coordinates = self.cut_out_coordinates.iloc[idx-1, :].values

        cut_out = get_cut_out(self.map, cut_out_coordinates, self.map_minmax_values)
        cut_out = torch.from_numpy(cut_out).float()

        #sample = np.concatenate((sweep, cut_out))
        sample = torch.cat((sweep, cut_out))

        training_sample = {'sample': sample, 'labels': labels}  # .values}  #  This worked on Sabinas Mac.

        return training_sample


def get_sweep_loaders(path_training_data, map_path, map_minmax_values_path, path_validation_data, batch_size, kwargs):
    csv_file = path_training_data + '/labels.csv'
    sample_dir = path_training_data + '/samples/'
    training_data_set = LiDARDataSet_onthego(csv_file, sample_dir, map_path, map_minmax_values_path)

    csv_file = path_validation_data + '/labels.csv'
    sample_dir = path_validation_data + '/samples/'
    validation_data_set = LiDARDataSet_onthego(csv_file, sample_dir, map_path, map_minmax_values_path)

    val_size = int(0.8 * len(validation_data_set))
    val_dataset = torch.utils.data.dataset.Subset(validation_data_set, np.arange(1, val_size+1))
    test_dataset = torch.utils.data.dataset.Subset(validation_data_set, np.arange(val_size+1, len(validation_data_set)+1))

    # Training
    n_training_samples = len(training_data_set)
    print('Number of training samples: ', n_training_samples)
    train_sampler = SubsetRandomSampler(np.arange(1, n_training_samples, dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(training_data_set, batch_size=batch_size, sampler=train_sampler, num_workers=2, **kwargs)

    # Validation
    n_val_samples = len(val_dataset)
    print('Number of validation samples: ', n_val_samples)
    val_sampler = SubsetRandomSampler(np.arange(n_val_samples, dtype=np.int64))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2, **kwargs)

    # Test
    n_test_samples = len(test_dataset)
    print('Number of test samples: ', n_test_samples)
    print(' ')
    test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=2, **kwargs)

    return train_loader, val_loader, test_loader
