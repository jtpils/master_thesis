from data_set_point_cloud import *
from lidar_processing_functions import *
from preprocessing_data_functions import *
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

use_cuda = False
batch_size = 1
data_set_path = '/home/master04/Desktop/Dataset/Town02_sorted_grid_ply'
number_of_samples = 1
workers = 0
kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {'num_workers': workers}
data_loader = get_train_loader(batch_size, data_set_path, number_of_samples, kwargs)


for i, data in enumerate(data_loader):
    print(i)
    sweep, map, labels = data['sweep'].numpy(), data['map'].numpy(), data['labels']
    sweep, map = np.squeeze(sweep, axis=0), np.squeeze(map, axis=0)
    pillar_dict = create_pillars(sweep, pillar_size=1)

list_keys = pillar_dict.keys()
for key in tqdm(list_keys):
    x = pillar_dict[key][:,0]
    y = pillar_dict[key][:,1]
    plt.plot(x,y,'.')
plt.show()

tensor = get_feature_tensor(pillar_dict, max_number_of_pillars=12000, max_number_of_points_per_pillar=100, dimension=8)

x = tensor[0,:,:]
y = tensor[1,:,:]
plt.plot(x,y,'.')
plt.show()



'''
number_samples = 2
batch_size = 1
print('Initializing data set: ')
data_path = '/home/master04/Desktop/Dataset/Town02_sorted_grid_ply'
dataset = PointCloudDataSet(data_set_path=data_path, number_of_samples=number_samples)
print('Done initializing data set. ')
print(' ')

print('Creating samples. ')
idx = 0
for idx in range(1):
    sample_dict = dataset.__getitem__(idx)
    print(np.shape(sample_dict['sweep']))
    print(np.shape(sample_dict['map']))
    print(' ')

'''





'''
print('discretizing sweep: ')
sweep = discretize_pointcloud(sample_dict['sweep'], array_size=60, trim_range=15, spatial_resolution=0.5, padding=False)
print(' ')
print('Creating png: ')
array_to_png(sweep, 'sweep_png')
del sweep

print('discretizing map: ')
cutout = discretize_pointcloud(sample_dict['map'], array_size=90, trim_range=22, spatial_resolution=0.5, padding=False)
print(' ')
print('Creating png: ')
array_to_png(cutout, 'map_png')
del cutout
'''

