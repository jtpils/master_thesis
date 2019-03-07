import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from LiDARDataSet import LiDARDataSet


def get_loaders(path_training_data, path_validation_data, batch_size, kwargs):
    csv_file = path_training_data + '/labels.csv'
    sample_dir = path_training_data + '/samples/'
    training_data_set = LiDARDataSet(csv_file, sample_dir)

    csv_file = path_validation_data + '/labels.csv'
    sample_dir = path_validation_data + '/samples/'
    validation_data_set = LiDARDataSet(csv_file, sample_dir)

    val_size = int(0.8 * len(validation_data_set))
    val_dataset = torch.utils.data.dataset.Subset(validation_data_set, np.arange(1, val_size+1))
    test_dataset = torch.utils.data.dataset.Subset(validation_data_set, np.arange(val_size+1, len(validation_data_set)+1))

    # Old stuff
    # lidar_data_set = LiDARDataSet(csv_file, sample_dir)
    # train_size = int(0.7 * len(lidar_data_set))
    # val_size = int(0.2 * len(lidar_data_set))
    # train_dataset = torch.utils.data.dataset.Subset(lidar_data_set, np.arange(1, train_size+1))
    # val_dataset = torch.utils.data.dataset.Subset(lidar_data_set, np.arange(train_size+1, train_size+val_size+1))
    # test_dataset = torch.utils.data.dataset.Subset(lidar_data_set, np.arange(train_size+val_size+1, len(lidar_data_set)+1))

    # Training
    n_training_samples = len(training_data_set)
    print('Number of training samples: ', n_training_samples)
    train_sampler = SubsetRandomSampler(np.arange(1, n_training_samples, dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(training_data_set, batch_size=batch_size, sampler=train_sampler, num_workers=4, **kwargs)

    # Validation
    n_val_samples = len(val_dataset)
    print('Number of validation samples: ', n_val_samples)
    val_sampler = SubsetRandomSampler(np.arange(n_val_samples, dtype=np.int64))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, **kwargs)

    # Test
    n_test_samples = len(test_dataset)
    print('Number of test samples: ', n_test_samples)
    test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, **kwargs)

    return train_loader, val_loader, test_loader
