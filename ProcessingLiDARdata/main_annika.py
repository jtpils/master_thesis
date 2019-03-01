import numpy as np
from lidar_data_functions import *
from map_functions import *
from matplotlib import pyplot as plt
import pandas as pd


path_to_ply = '/home/master04/Desktop/_out_town2/pc/059176.ply'
path_to_csv = '/home/master04/Desktop/_out_town2/town2.csv'


point_cloud, global_coordinates = load_data(path_to_ply, path_to_csv)

# rotate, translate the point cloud to global coordinates and trim the point cloud
trimmed_pc = trim_pointcloud(point_cloud, range=20, roof=100, floor=0.5)

rotated_pc = rotate_pointcloud_to_global(trimmed_pc, global_coordinates)

rotated_and_translated_pc = translate_pointcloud_to_global(rotated_pc, global_coordinates)

# global_coordinates_plot = np.loadtxt(path_to_csv, skiprows=1, delimiter=',')
global_coordinates_plot = pd.read_csv(path_to_csv)
global_coordinates_plot = global_coordinates_plot.values
global_coordinates_plot[:,2] = - global_coordinates_plot[:,2]

x = rotated_and_translated_pc[:, 0]
y = rotated_and_translated_pc[:, 1]
#print(x)

plt.plot(x , y,  'b.')
plt.plot(global_coordinates_plot[:, 1],global_coordinates_plot[:, 2],'kd')
plt.ylabel('y')
plt.xlabel('x')
plt.show()


