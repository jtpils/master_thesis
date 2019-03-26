from data_set_point_cloud import *
from lidar_processing_functions import *
from preprocessing_data_functions import *
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from train_network import *

path_data_set = '/home/master04/Desktop/Dataset/point_cloud/pc_small_set'
#file = open(path_data_set + '/training_sample_1', 'rb')
#training_sample = pickle.load(file)

#sweep = training_sample['sweep']

train_loader = get_train_loader_pc(1, path_data_set, 2, {})

train_loss, val_loss = train_network(1, 0.001, 1, folder_path, False, batch_size)

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

