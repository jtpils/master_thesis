import numpy as np
from lidar_data_functions import *
import random
from PIL import Image
import time
import matplotlib.pyplot as plt
import os
import sys
import math


# TODO: check that it works with the spatial resolution = 0.05.
#  Test to use a discretized map and do a cut out that is 900x900 (show the map and the cut out).
#  Annika code check

def rounding(n, r=0.05):
    '''
    Function for round down to nearest r.

    :param n: Number that is goint to be round
    :param r: The number that we want to round to Now it works only with 0.05
    :return: rounded_number: The value of the rounded number
    '''

    if n >= 0:
        rounded_number = round(n - math.fmod(n, r), 2)
    elif n < 0:
        n = -n + (r + r/5) # The + (r + r/5) is for get the right interval when having negative number
        rounded_number = -round(n - math.fmod(n, r), 2)
    return rounded_number


def get_cut_out(discretized_point_cloud_map, global_coordinate, max_min_values_map, spatial_resolution=0.05,
                cut_out_size=900):
    '''
    Function that creates a cut out from the discretized map.

    :param discretized_point_cloud_map: (np.array) The discretized point cloud map, the map is quadratic.
    :param global_coordinate: (np.array) The global coordinate representing the initial guess.
    :param max_min_values_map: (np.array) The maximum and minimum values map defining the corners of the map. (This should be an output from the discretize map function
    :param spatial_resolution: (int) The spatial resolution of the map, how large a cell is. , should be an output from the map.
    :param cut_out_size: (int) The size of the cut_out. The cut_out should be quadratic, e.g 900x900
    :return: cut_out: (np.array) A cut out of the map at the initial guess.
    '''
    x_min, x_max, y_min, y_max = max_min_values_map

    # check if we have a global coordinate that is outside the map coordinates. If that's the case stop execution.
    if global_coordinate[0] < x_min or x_max < global_coordinate[0] or global_coordinate[1] < y_min or y_max < \
            global_coordinate[1]:
        print('Global coordinate,', global_coordinate, ', is located outside the map boundaries.')

        print('Maximum x value:', x_max, '. Minimum x value:', x_min, '. Maximum y value:', y_max, '. Minimum y value:',
              y_min)

        print(' ')
        return None
        #sys.exit(0)  # we do not want to exit we just want to brake try exept. leave the function!

    # PAD THE MAP HERE!
    # Things that need to be considered is if padding the map here the new bounds must be considered!!!
    # The padding is performed by adding image//2-1 zeros below column 0 and image//2 zeros after last column.
    # image//2-1 zeros below row 0 and image//2 zeros after last row.

    pad_size_low = cut_out_size // 2 - 1
    pad_size_high = cut_out_size // 2

    # print('shape of map befor padding',np.shape(discretized_point_cloud_map))
    discretized_point_cloud_map = np.pad(discretized_point_cloud_map,
                                         [(0, 0), (pad_size_low, pad_size_high), (pad_size_low, pad_size_high)],
                                         mode='constant')
    # print('shape of map after padding', np.shape(discretized_point_cloud_map))

    # Check the x value and "put" it in the right cell of the map
    # Bounds must be fixed since the map now is bigger and have zeros! Lower bound of x is now (x_min - pad_size_low)
    # and Upper bound of x is (x_max + pad_size_high)

    cell_check_x = rounding(x_min, spatial_resolution) - pad_size_low*spatial_resolution
    # print('start_cell_check', cell_check_x)
    k = 0  # sice python starts at 0
    while cell_check_x < global_coordinate[0]:
        x_cell = k
        k += 1
        cell_check_x += spatial_resolution
    # print('x_cell: ', x_cell)

    # Check the y value and "put" it in the right cell of the map
    # Bounds must be fixed since the map now is bigger and have zeros! Lower bound of y is now (y_min - pad_size_low)
    # and Upper bound of y is (y_max + pad_size_high)

    cell_check_y = rounding(y_min, spatial_resolution) - pad_size_low*spatial_resolution
    # print('start_cell_check', cell_check_y)
    k = 0  # since python starts at 0
    while cell_check_y < global_coordinate[1]:
        y_cell = k
        k += 1
        cell_check_y += spatial_resolution
    # print('y_cell: ', y_cell)

    # start to find deviation how to cut the map and then cut out a piece.
    deviation_from_cell_low = cut_out_size // 2 - 1
    deviation_from_cell_high = cut_out_size // 2

    # print('deviation from cell low:', deviation_from_cell_low)
    # print('deviation from cell high:', deviation_from_cell_high)

    # variable name change for clarifying row and column bounds.
    col_cell = x_cell
    row_cell = y_cell

    lower_bound_row = row_cell - deviation_from_cell_low
    upper_bound_row = row_cell + deviation_from_cell_high + 1  # +1 to include upper bound

    lower_bound_col = col_cell - deviation_from_cell_low
    upper_bound_col = col_cell + deviation_from_cell_high + 1  # +1 to include upper bound

    # print('low bound row:', lower_bound_row)
    # print('high bound row:', upper_bound_row)

    # print('low bound col:', lower_bound_col)
    # print('high bound col:', upper_bound_col)

    # Create the cut out
    cut_out = discretized_point_cloud_map[:, lower_bound_row:upper_bound_row, lower_bound_col:upper_bound_col]

    return cut_out

'''
# Test the function with this section. 
for i in range(1):  # range sets how many cut_outs to do.

    # load the global coordinates
    path_to_csv = '/home/master04/Desktop/_out_Town02_190221_1/Town02_190221_1.csv'
    global_coordinates_pc = pd.read_csv(path_to_csv)
    global_coordinates_pc = global_coordinates_pc.values
    # load the map
    discretized_pc_map = np.load('/home/master04/Documents/master_thesis/ProcessingLiDARdata/map_190304_testing/map.npy')

    #load the max min values
    max_min_values_map = np.load('/home/master04/Documents/master_thesis/ProcessingLiDARdata/map_190304_testing/max_min.npy')
    print( 'x_min:', max_min_values_map[0],'x_max:', max_min_values_map[1], 'y_min:', max_min_values_map[2], 'y_max:', max_min_values_map[3])

    # take a random global coordinate
    row = random.randint(0, len(global_coordinates_pc))
    print('row', row)
    global_coordinates = global_coordinates_pc[row,1:4]
    global_coordinates[1] = -global_coordinates[1]

    print('global coordninates: ', global_coordinates)


    # Since the spatial resolution of the map in the testing is 0.5 i.e 10 times smaller than the real map is going to
    # be the cut out must be 10 times smaller as well, there of 90 in stead of 900
    cut_out = get_cut_out(discretized_pc_map, global_coordinates, max_min_values_map, spatial_resolution=0.05, cut_out_size=900)
    print('shape of the cut_out: ' , np.shape(cut_out))

    # VISUALIZATION OF DISCRETIZED CUT OUT
    layer = 2
    max_value = np.max(cut_out[layer, :, :])
    print('Max max_value in array_to_png: ', max_value)

    # avoid division with 0
    if max_value == 0:
        max_value = 1

    scale = 255/max_value
    cut_out[layer, :, :] = cut_out[layer, :, :] * scale
    print('Largest pixel value (should be 255) : ', np.max(cut_out[layer, :, :]))

    img = Image.fromarray(cut_out[layer, :, :])
    new_img = img.convert("L")
    new_img.rotate(180).show()


# VISUALIZATION OF DISCRETIZED MAP
# normalize the BEV image
layer = 2
max_value = np.max(discretized_pc_map[layer, :, :])
print('Max max_value in array_to_png: ', max_value)

# avoid division with 0
if max_value == 0:
    max_value = 1

scale = 255/max_value
discretized_pc_map[layer, :, :] = discretized_pc_map[layer, :, :] * scale
print('Largest pixel value (should be 255) : ', np.max(discretized_pc_map[layer, :, :]))

img = Image.fromarray(discretized_pc_map[layer, :, :])
new_img = img.convert("L")
new_img.rotate(180).show()

'''
