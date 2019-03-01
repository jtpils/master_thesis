import numpy as np
from lidar_data_functions import *
import random
from PIL import Image
import time
import matplotlib.pyplot as plt
import os
import sys
import random
import math


# TODO: check that it works with the spatial resolution = 0.05.
#  Change in discretized map: return the max and min values of the map.
#  Test to use a discretized map and do a cut out that is 900x900 (show the map and the cut out).
#  Annika code check

def rounding(n, r=0.05):
    '''

    :param n: Number that is goint to be round
    :param r: The number that we want to round to Now it works only with 0.05
    :return: rounded_number: The value of the rounded number
    '''

    if n >= 0:
        rounded_number = round(n - math.fmod(n, r),2)
        print(rounded_number)

    elif n < 0:
        n = -n + 0.06  # The + 0.06 is for get the
        rounded_number = -round(n - math.fmod(n, r),2)
        print(rounded_number)

    return rounded_number



def get_cut_out(discretized_point_cloud_map, global_coordinate, max_min_values_map, spatial_resolution=0.05, cut_out_size=900):
    '''
    :param discretized_point_cloud_map: (np.array) The discretized point cloud map, the map is quadratic.
    :param global_coordinate: (np.array) The global coordinate representing the initial guess.
    :param max_min_values_map: (np.array) The maximum and minimum values map defining the corners of the map. (This should be an output from the discretize map function
    :param spatial_resolution: (int) The spatial resolution of the map, how large a cell is. , should be an output from the map.
    :param cut_out_size: (int) The size of the cut_out. The cut_out should be quadratic, e.g 900x900
    :return: cut_out: (np.array) A cut out of the map at the initial guess.
    '''

    x_min, x_max, y_min, y_max = max_min_values_map
    print('x_max:', x_max, 'x_min:', x_min, 'y_min:', y_min, 'y_max:', y_max)

    # check if we have a global coordinate that is outside the map coordinates. If that's the case stop execution.
    if global_coordinate[0] < x_min or x_max < global_coordinate[0] or global_coordinate[1] < y_min or y_max < global_coordinate[1]:
        print('Global coordinate,', global_coordinate, ', is located outside the map boundaries.')
        print(' ')
        print('Maximum x value:', x_max, '. Minimum x value:', x_min, '. Maximum y value:', y_max, '. Minimum y value:', y_min)
        sys.exit(0) # we do not want to exit we just want to brake try exept. leave the function!

    # PAD THE MAP HERE!
    # Things that need to be considered is if padding the map here the new bounds must be considered!!!
    # The padding is performed by adding image//2-1 zeros below column 0 and image//2 zeros after last column. image//2-1 zeros below row 0 and image//2 zeros after last row.

    pad_size_low = cut_out_size//2 - 1
    pad_size_high = cut_out_size//2

    print('pad size low', pad_size_low)
    print('pad size high', pad_size_high)

    discretized_point_cloud_map = np.pad(discretized_point_cloud_map, [(0, 0), (pad_size_low, pad_size_high), (pad_size_low, pad_size_high)], mode='constant')

    #print('discretized pc map:', discretized_point_cloud_map)

    # here I want to check which cell i want to cut out.
    # Check the x value and "put" it in the right cell of the map
    # Now the bounds must be fixed since the map now is bigger and have zeros! Lower bound of x is now (x_min - pad_size_low) and Upper bound of x is (x_max + pad_size_high)

    cell_check_x = rounding(x_min) - pad_size_low
    print('start_cell_check', cell_check_x)
    #cell_check_x = (int(np.floor(x_min)) - pad_size_low) # we want to start at the nearest sppatial resolution.
    k = 0 # sice python starts at 0
    while cell_check_x < global_coordinate[0]:
        #print('cell_location: ', cell_location)
        x_cell = k
        k += 1
        cell_check_x += spatial_resolution
        print('cell_check:', cell_check_x, ', x_cell:', x_cell)

    print('x_cell: ', x_cell)

    # Check the y value and "put" it in the right cell of the map
    # Now the bounds must be fixed since the map now is bigger and have zeros! Lower bound of y is now (y_min - pad_size_low) and Upper bound of y is (y_max + pad_size_high)

    #cell_check_y = int(np.floor(y_min)) - pad_size_low #
    cell_check_y = rounding(y_min) - pad_size_low
    print('start_cell_check', cell_check_y)
    k = 0 # sice python starts at 0
    while cell_check_y < global_coordinate[1]:
        #print('cell_location: ', cell_location)
        y_cell = k
        k += 1
        cell_check_y += spatial_resolution
        print('y_cell_check:', cell_check_y, ', y_cell:', y_cell)

    print('y_cell: ', y_cell)


    # start to find deviation how to cut the map and then cut out a piece. 
    
    deviation_from_cell_low = cut_out_size//2 - 1
    deviation_from_cell_high = cut_out_size//2 
    
    print('deviation from cell low:', deviation_from_cell_low)
    print('deviation from cell high:', deviation_from_cell_high)

    # for clarifying rox and column bounds.
    row_cell = y_cell
    col_cell = x_cell

    lower_bound_row = row_cell - deviation_from_cell_low
    upper_bound_row = row_cell + deviation_from_cell_high + 1  # +1 to include upper bound

    lower_bound_col = col_cell - deviation_from_cell_low
    upper_bound_col = col_cell + deviation_from_cell_high + 1 # +1 to include upper bound

    print('low bound row:' , lower_bound_row)
    print('high bound row:' , upper_bound_row)

    print('low bound col:', lower_bound_col)
    print('high bound col:', upper_bound_col)

    print(discretized_point_cloud_map[0,:,:])

    # Do the cut out
    cut_out = discretized_point_cloud_map[:, lower_bound_row:upper_bound_row, lower_bound_col:upper_bound_col]

    print(np.shape(cut_out))

    print(cut_out[0,:,:])

    return cut_out


# TIME TO TEST ON A REAL MAP!

'''
max_min_values_map = np.array((0, 1, 0, 1))
x_min, x_max, y_min, y_max = max_min_values_map
spatial_resolution = 0.05
number_x_cells = int(np.ceil((x_max - x_min) / spatial_resolution))  # should there be a +1 or -1 or something like that? Now Sabina has set 10 of some reason she can't explain.
number_y_cells = int(np.ceil((y_max - y_min) / spatial_resolution))  # should there be a +1 or -1 or something like that?

discretize_pointcloud_map = np.zeros([4, number_x_cells, number_y_cells])
print(np.shape(discretize_pointcloud_map))
# map 10x10
#discretize_pointcloud_map[:,0:2,:] = 1
#discretize_pointcloud_map[:,2:4,:] = 2
#discretize_pointcloud_map[:,4:6,:] = 3
#discretize_pointcloud_map[:,6:8,:] = 4
#discretize_pointcloud_map[:,8:11,:] = 5

# map 5x5
discretize_pointcloud_map[:,0:4,:] = 1
discretize_pointcloud_map[:,4:8,:] = 2
discretize_pointcloud_map[:,8:10,:] = 3

cut_out_size = 6

global_coordinate = [0.01,0.63, 3]  # x,y,z
#global_coordinate = [3.35, 8.32, 3.2, 1.3, 0.2, 0.0, 10]
cut_out = get_cut_out(discretize_pointcloud_map, global_coordinate, max_min_values_map, spatial_resolution, cut_out_size)
'''
