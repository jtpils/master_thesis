import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from LiDARDataSet import LiDARDataSet


def get_loaders(path_training_data, batch_size_train, batch_size_val):
    csv_file = path_training_data + '/labels.csv'
    sample_dir = path_training_data + '/samples/'

    lidar_data_set = LiDARDataSet(csv_file, sample_dir)

    # Training
    n_training_samples = np.ceil(0.7*len(lidar_data_set))
    print('Number of training samples: ', n_training_samples)
    train_sampler = SubsetRandomSampler(np.arange(1, n_training_samples, dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(lidar_data_set, batch_size=batch_size_train, sampler=train_sampler, num_workers=4)

    # Validation
    n_val_samples = np.floor(0.3*len(lidar_data_set))
    print('Number of validation samples: ', n_val_samples)
    val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))
    val_loader = torch.utils.data.DataLoader(lidar_data_set, batch_size=batch_size_val, sampler=val_sampler, num_workers=4)

    return train_loader, val_loader
