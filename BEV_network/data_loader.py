import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from LiDARDataSet import LiDARDataSet


def get_loaders(path_training_data, path_validation_data, batch_size, use_cuda):
    csv_file = path_training_data + '/labels.csv'
    sample_dir = path_training_data + '/samples/'
    training_data_set = LiDARDataSet(csv_file, sample_dir, use_cuda)

    csv_file = path_validation_data + '/labels.csv'
    sample_dir = path_validation_data + '/samples/'
    validation_data_set = LiDARDataSet(csv_file, sample_dir, use_cuda)

    kwargs = {'pin_memory': True} if use_cuda else {}
    workers_train = 16
    workers_val = 4
    print('Number of workers: ', workers_train)

    # Training
    n_training_samples = len(training_data_set)
    print('Number of training samples: ', n_training_samples)
    train_sampler = SubsetRandomSampler(np.arange(1, n_training_samples+1, dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(training_data_set, batch_size=batch_size, sampler=train_sampler, num_workers=workers_train, **kwargs)

    # Validation
    n_val_samples = len(validation_data_set)
    print('Number of validation samples: ', n_val_samples)
    val_sampler = SubsetRandomSampler(np.arange(1, n_val_samples+1, dtype=np.int64))
    val_loader = torch.utils.data.DataLoader(validation_data_set, batch_size=batch_size, sampler=val_sampler, num_workers=workers_val, **kwargs)

    print(' ')

    return train_loader, val_loader
