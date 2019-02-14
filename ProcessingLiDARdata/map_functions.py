# here we will add our functions that are specific for the map
import numpy as np
from PIL import Image
import os
import sys

def discretize_pointcloud_map(map_point_cloud, min_max_coordinates, spatial_resolution=0.50):
    '''
    Discretize into a grid structure with different channels.
    Create layers:
    1 layer with number of detections (normalise at the end), 1 layer with mean height, max height, min height. other layers?
    :param
        pointcloud_dict:
        min_max_coordinates: array with [xmin, xmax, ymin, ymax]
        spatial_resolution: size of grid cell in m
    :return:
        discretized_pointcloud: 3d-array with multiple "layers"
        layer 0 = number of detections
        layer 1 = mean evaluation
        layer 2 = maximum evaluation
        layer 3 = minimum evaluation
    '''

    # calculate the dimension of the array needed to store all the loaded ply-files
    x_min, x_max, y_min, y_max = min_max_coordinates
    number_x_cells = int(np.ceil((x_max - x_min) / spatial_resolution)) +10 # should there be a +1 or -1 or something like that? Now Sabina has set 10 of some reason she can't explain.
    number_y_cells = int(np.ceil((y_max - y_min) / spatial_resolution)) +10# should there be a +1 or -1 or something like that?
    # print('number of cells', number_x_cells)

    discretized_pointcloud = np.zeros([4, number_x_cells, number_y_cells])

    # sort the point cloud by x in increasing order
    x_sorted_point_cloud = np.asarray(sorted(map_point_cloud, key=lambda row: row[0]))

    for x_cell in range(number_x_cells):

        # get the x-values in the spatial resolution interval
        lower_bound = x_min + spatial_resolution * x_cell
        upper_bound = x_min + (x_cell + 1) * spatial_resolution
        x_interval = list(map(lambda x: lower_bound < x <= upper_bound, x_sorted_point_cloud[:, 0]))

        x_interval = x_sorted_point_cloud[x_interval]

        # sort the x-interval by increasing y
        x_sorted_by_y = np.asarray(sorted(x_interval, key=lambda row: row[1]))

        # loop through the y coordinates in the current x_interval and store values in the output_channel
        for y_cell in range(number_y_cells):

            # if len(sorted_y) is 0:
            if len(x_sorted_by_y) is 0:
                discretized_pointcloud[0, x_cell, y_cell] = 0
            else:
                lower_bound = y_min + spatial_resolution * y_cell
                upper_bound = y_min + (y_cell + 1) * spatial_resolution
                y_interval = np.asarray(x_sorted_by_y[list(map(lambda x: lower_bound < x <= upper_bound, x_sorted_by_y[:, 1]))])

                # if there are detections save these in right channel
                if np.shape(y_interval)[0] is not 0:
                    discretized_pointcloud[0, x_cell, y_cell] = np.shape(y_interval)[0]
                    discretized_pointcloud[1, x_cell, y_cell] = np.mean(y_interval[:, 2])
                    discretized_pointcloud[2, x_cell, y_cell] = np.max(y_interval[:, 2])
                    discretized_pointcloud[3, x_cell, y_cell] = np.min(y_interval[:, 2])

                # if there are not any detections
                else:
                    discretized_pointcloud[0, x_cell, y_cell] = 0

    return discretized_pointcloud
