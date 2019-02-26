import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from LiDARDataSet import LiDARDataSet


def get_loaders(path_training_data, batch_size_train, batch_size_val, kwargs, train_split=0.7):
    csv_file = path_training_data + '/labels.csv'
    sample_dir = path_training_data + '/samples/'

    lidar_data_set = LiDARDataSet(csv_file, sample_dir)

    train_size = int(0.7 * len(lidar_data_set))
    val_size = int(0.2 * len(lidar_data_set))
    #test_size = len(lidar_data_set) - train_size - val_size

    train_dataset = torch.utils.data.dataset.Subset(lidar_data_set, np.arange(1, train_size+1))
    val_dataset = torch.utils.data.dataset.Subset(lidar_data_set, np.arange(train_size+1, train_size+val_size+1))
    test_dataset = torch.utils.data.dataset.Subset(lidar_data_set, np.arange(train_size+val_size+1, len(lidar_data_set)+1))

    # Training
    n_training_samples = len(train_dataset)
    print('Number of training samples: ', n_training_samples)
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, sampler=train_sampler, num_workers=4, **kwargs)

    # Validation
    n_val_samples = len(val_dataset)
    print('Number of validation samples: ', n_val_samples)
    val_sampler = SubsetRandomSampler(np.arange(n_val_samples, dtype=np.int64))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, sampler=val_sampler, num_workers=4, **kwargs)

    # Test
    n_test_samples = len(test_dataset)
    print('Number of test samples: ', n_test_samples)
    test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_val, sampler=test_sampler, num_workers=4, **kwargs)

    return train_loader, val_loader, test_loader
