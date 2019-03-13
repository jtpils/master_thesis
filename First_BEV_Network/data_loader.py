import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from LiDARDataSet import LiDARDataSet


def get_loaders(path_training_data, path_validation_data, path_test_data, batch_size, use_cuda, kwargs):
    csv_file = path_training_data + '/labels.csv'
    sample_dir = path_training_data + '/samples/'
    training_data_set = LiDARDataSet(csv_file, sample_dir, use_cuda)

    csv_file = path_validation_data + '/labels.csv'
    sample_dir = path_validation_data + '/samples/'
    validation_data_set = LiDARDataSet(csv_file, sample_dir, use_cuda)

    csv_file = path_test_data + '/labels.csv'
    sample_dir = path_test_data + '/samples/'
    test_data_set = LiDARDataSet(csv_file, sample_dir, use_cuda)

    # Training
    n_training_samples = 200 #len(training_data_set)
    print('Number of training samples: ', n_training_samples)
    train_sampler = SubsetRandomSampler(np.arange(1, n_training_samples+1, dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(training_data_set, batch_size=batch_size, sampler=train_sampler, num_workers=4, **kwargs)

    # Validation
    n_val_samples = 20 #len(validation_data_set)
    print('Number of validation samples: ', n_val_samples)
    val_sampler = SubsetRandomSampler(np.arange(1, n_val_samples+1, dtype=np.int64))
    val_loader = torch.utils.data.DataLoader(validation_data_set, batch_size=batch_size, sampler=val_sampler, num_workers=4, **kwargs)

    # Test
    n_test_samples = len(test_data_set)
    print('Number of test samples: ', n_test_samples)
    test_sampler = SubsetRandomSampler(np.arange(1, n_test_samples+1, dtype=np.int64))
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, sampler=test_sampler, num_workers=4, **kwargs)

    print(' ')

    return train_loader, val_loader, test_loader
