from data_set_point_cloud import *
from lidar_processing_functions import *

print('Initializing data set: ')
data_path = '/Users/annikal/Desktop/Ply_files/TEST_sorted_grid_ply'
dataset = PointCloudDataSet(data_set_path=data_path, number_of_samples=1)
print('Done initializing data set. ')
print(' ')

print('Creating samples. ')
idx = 0
sample_dict = dataset.__getitem__(idx)

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
