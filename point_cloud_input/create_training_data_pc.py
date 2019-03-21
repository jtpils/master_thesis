import numpy as np
from data_set_point_cloud import *
from preprocessing_data_functions import *
import os
from tqdm import tqdm

# TODO: 1. Iterate over data loader. 2. for the data take the sweep and cutout and make two tensors of them and save
#  them together with the label.


input_folder_name = input('Type name of new folder to save data:')
print(' ')

current_path = os.getcwd()

folder_path = os.path.join(current_path, input_folder_name)

try:
    os.mkdir(folder_path)
except OSError:
    print('Failed to create new directory.')
else:
    print('Successfully created new directory with subdirectory, at ', folder_path)


batch_size = 1
path_to_lidar_data = '/Users/sabinalinderoth/Desktop/Ply_files/TEST_sorted_grid_ply/' #input('Type complete path to the folder that contains "pc"-folder with LiDAR data and a csv file, e.g. in the Ply_files/sorted_grid :')
number_of_samples = 10 #int(input('Type number of samples to create:'))

train_loader = get_train_loader(batch_size, path_to_lidar_data, number_of_samples, {})


for i, data in tqdm(enumerate(train_loader, 1)):
    
    print('Creating training sample ', i, ' of ', len(train_loader))
    
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

    #print(np.shape(map_tensor))

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

    file_name = 'training_sample_' + str(i) + '.npy'

    sample_path = os.path.join(folder_path,file_name)

    np.save(sample_path, training_sample)

