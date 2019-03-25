import numpy as np
from data_set_point_cloud import *
from preprocessing_data_functions import *
import os
from tqdm import tqdm
import pickle


input_folder_name = input('Type name of new folder to save data:')
current_path = os.getcwd()
folder_path = os.path.join(current_path, input_folder_name)

try:
    os.mkdir(folder_path)
except OSError:
    print('Failed to create new directory.')
else:
    print('Successfully created new directory with subdirectory, at ', folder_path)

batch_size = 1
path_to_lidar_data = input('Type complete path to the folder that contains grid-folders with LiDAR data and a csv file, '
                           'e.g. in the Ply_files/sorted_grid :')

number_of_samples = int(input('Type number of samples to create:'))

train_loader = get_train_loader(batch_size, path_to_lidar_data, number_of_samples, {})

for i, data in tqdm(enumerate(train_loader, 1)):
    
    print('Creating training sample ', i, ' of ', len(train_loader))

    map_cutout = data['map']

    # convert tensors to np.arrays and sqeeze to change size from (1,x,y) to (x,y).
    map_cutout = map_cutout.numpy()
    map_cutout = np.squeeze(map_cutout, axis=0)

    # create pillars
    map_pillars = create_pillars(map_cutout, pillar_size=0.16)

    # get the feature tensor
    map_tensor = get_feature_tensor(map_pillars, max_number_of_pillars=12000, max_number_of_points_per_pillar=100,
                                    dimension=8)

    # Do the same for the sweep
    sweep = data['sweep']
    sweep = sweep.numpy()
    sweep = np.squeeze(sweep, axis=0)

    sweep_pillars = create_pillars(sweep, pillar_size=0.16)
    sweep_tensor = get_feature_tensor(sweep_pillars, max_number_of_pillars=12000, max_number_of_points_per_pillar=100,
                                    dimension=8)

    labels = data['labels']
    labels = labels.numpy()

    # Save the map cutout and the sweep in a folder together with the labels
    training_sample = {'sweep': sweep_tensor, 'map': map_tensor, 'labels': labels}

    file_name = 'training_sample_' + str(i)

    sample_path = os.path.join(folder_path,file_name)

    # Save the sample
    pickle_out = open(sample_path, "wb")
    pickle.dump(training_sample, pickle_out)
    pickle_out.close()
    print('')
    print('Created sample: ', i)

    '''
    code to load the file later: 
    file_name = 'type path to file'
    pickle_in = open(file_name, "rb")
    dict_sample = pickle.load(pickle_in)
    '''
