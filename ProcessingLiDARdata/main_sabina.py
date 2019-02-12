import numpy as np
from lidar_data_functions import *
from PIL import Image
import time
import matplotlib.pyplot as plt

path_to_ply = '/home/master04/Desktop/_out/empty_files/169244.ply'
path_to_csv = '/home/master04/Desktop/_out/_out_Town02_190208_1/Town02_190208_1.csv'

# Load data:
pc, global_lidar_coordinates = load_data(path_to_ply, path_to_csv)
print('Shape of raw pointcloud: ',np.shape(pc))
