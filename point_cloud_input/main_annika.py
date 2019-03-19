from data_set_point_cloud import *
from lidar_processing_functions import *
import numpy as np
import pandas as pd
import time


number_samples = 20
batch_size = 2
print('Initializing data set: ')
data_path = '/home/master04/Desktop/Dataset/Town02_sorted_grid_ply'
dataset = PointCloudDataSet(data_set_path=data_path, number_of_samples=number_samples)
print('Done initializing data set. ')
print(' ')
'''
print('Creating samples. ')
idx = 0
t1 = time.time()
for idx in range(10):
    sample_dict = dataset.__getitem__(idx)
    print(np.shape(sample_dict['sweep']))
    print(np.shape(sample_dict['map']))
    print(' ')
t2 = time.time()

print('Time: ', t2-t1)
'''


train_loader = get_train_loader(batch_size, data_path, number_samples, kwargs={})



print('Initializing data set: ')
data_path = '/Users/sabinalinderoth/Desktop/Ply_files/TEST_sorted_grid_ply'



dataset = PointCloudDataSet(data_set_path=data_path, number_of_samples=100)
print('Done initializing data set. ')
print(' ')

print('Creating samples. ')
#idx = 0
t1 = time.time()
for idx in tqdm(range(0,100)):

    sample_dict = dataset.__getitem__(idx)

t2 = time.time()
print('Time:', t2-t1)

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

