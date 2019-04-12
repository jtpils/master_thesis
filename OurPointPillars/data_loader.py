from DataSetPointPillars import *
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_loader(batch_size, data_set_path_train, csv_path_train, grid_csv_path_train, data_set_path_val,
                     csv_path_val, grid_csv_path_val, translation, rotation, kwargs):

    training_data_set = DataSetPointPillars(data_set_path_train, csv_path_train, grid_csv_path_train, translation=translation, rotation=rotation)
    print('Number of training samples: ', len(training_data_set))
    train_sampler = SubsetRandomSampler(np.arange(len(training_data_set), dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(training_data_set, batch_size=batch_size, sampler=train_sampler, drop_last = True, **kwargs)

    validation_data_set = DataSetPointPillars(data_set_path_val, csv_path_val, grid_csv_path_val, translation=translation, rotation=rotation)
    print('Number of training samples: ', len(validation_data_set))
    val_sampler = SubsetRandomSampler(np.arange(len(validation_data_set), dtype=np.int64))
    val_loader = torch.utils.data.DataLoader(validation_data_set, batch_size=batch_size, sampler=val_sampler, drop_last = True, **kwargs)

    return train_loader, val_loader
