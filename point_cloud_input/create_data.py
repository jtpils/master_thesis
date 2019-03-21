import numpy as np
from data_set_point_cloud import *
from preprocessing_data_functions import *
import os

# TODO:
# 1. Iterate over data loader
# 2. for the data take the sweep and cutout and make two tensors of them and save them together with the label.

folder_where_to_save_data = input('Enter folder name where to save data:')

batch_size = 1
data_set_path = '/Users/sabinalinderoth/Desktop/Ply_files/TEST_sorted_grid_ply/' #input('Enter path to where the data set (the one with the sorted grid folder) is located:')
number_of_samples = 1 #input('Enter number of samples to create:')

train_loader = get_train_loader(batch_size, data_set_path, number_of_samples, {})


for i, data in enumerate(train_loader, 1):

    map_cutout = data['map']
    sweep = data['sweep']
    labels = data['labels']

    # convert tensors to np.arrays.

    map_cutout = map_cutout.numpy()
    map_cutout = np.squeeze(map_cutout, axis=0)

    # create pillars
    map_pillars = create_pillars(map_cutout, pillar_size=0.16)
    # get the feature tensor
    map_tensor = get_feature_tensor(map_pillars, max_number_of_pillars=12000, max_number_of_points_per_pillar=100, dimension=8)

    print(np.shape(map_tensor))

    sweep = sweep.numpy()
    sweep = np.squeeze(sweep, axis=0)

    # create pillars
    sweep_pillars = create_pillars(sweep, pillar_size=0.16)
    # get the feature tensor
    sweep_tensor = get_feature_tensor(sweep_pillars, max_number_of_pillars=12000, max_number_of_points_per_pillar=100,
                                    dimension=8)


    labels = labels.numpy()
    # Do not know if i need to sqeeze this.
    labels = np.squeeze(labels, axis=0)

    # Save the map cutout and the sweep in a folder together with the label
    training_sample = {'sweep': sweep_tensor, 'map': map_tensor, 'labels': labels}

    file_name = 'training_sample' + str(i) + '.npy'

    path = os.path.join(folder_where_to_save_data,file_name)

    np.save(folder_where_to_save_data, training_sample)







