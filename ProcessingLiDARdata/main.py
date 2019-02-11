import numpy as np
from lidar_data_functions import *

path_to_ply = '/home/master04/Desktop/_out/_out_Town03_190207_17/pc/169244.ply'
path_to_csv = '/home/master04/Desktop/_out/_out_Town03_190207_17/Town03_190207_17.csv'

pointcloud, global_lidar_coordinates = load_data(path_to_ply, path_to_csv)

#print(global_lidar_coordinates)

grid_size = 0.05
point_cloud = np.array([[-15, 3, 3], [26, 2, 3], [1, 2, 3], [10, 28, 9], [4, 1, -4], [4, 1, 5], [11, 9, 1], [12, 1, 10],
                        [4, 8, 10], [1, 2, 3], [5, 7, 9], [25, 9, 1], [23, 3, 10]])


'''
point_cloud = np.array([[0.05, 0.05, -1], [0.05, 0.05, 12], [0.05, 0.05, 1],
                        [1, 11, -1], [1, 11, 12], [1, 11, 1],
                        [1, 21, -1], [1, 21, 12], [1, 21, 1],
                        [11, 1, -1], [11, 1, 12], [11, 1, 1],
                        [11, 11, -1], [11, 11, 12], [11, 11, 1],
                        [11, 21, -1], [11, 21, 12], [11, 21, 1],
                        [21, 1, -1], [21, 1, 12], [21, 1, 1],
                        [21, 11, -1], [21, 11, 12], [21, 11, 1],
                        [21, 21, -1], [21, 21, 12], [21, 21, 1]])

#point_cloud = np.zeros((9, 3))
#point_cloud = []

'''


def discretize_pointcloud(trimmed_point_cloud, spatial_resolution):
    '''
    Discretize into a grid structure with different channels.
    Create layers:
    1 layer with number of detections (normalise at the end), 1 layer with mean height, max height, min height. other layers?
    :param
        pointcloud:
        spatial_resolution: size of grid cell in m
    :return:
        discretized_pointcloud: 3d-array with multiple "layers"
        layer 0 = number of detections
        layer 1 = mean evaluation
        layer 2 = maximum evaluation
        layer 3 = minimum evaluation
    '''

    discretized_pointcloud = np.zeros([4, 3,3])

    if len(point_cloud) is 0:

        discretized_pointcloud[0, :, :] = 0
        discretized_pointcloud[1, :, :] = 'NaN'
        discretized_pointcloud[2, :, :] = 'NaN'
        discretized_pointcloud[3, :, :] = 'NaN'

    else:

        # sort the point cloud by x in increasing order
        x_sorted_point_cloud = np.asarray(sorted(point_cloud, key=lambda row: row[0]))

        for x_cell in range(3):

            # get the x-values in the spatial resolution interval
            x_interval = list(map(lambda x: (spatial_resolution * x_cell) < x <= (x_cell + 1) * spatial_resolution, x_sorted_point_cloud[:, 0]))
            x_interval = x_sorted_point_cloud[x_interval]

            # sort the x-interval by increasing y
            x_sorted_by_y = np.asarray(sorted(x_interval, key=lambda row: row[1]))

            # loop through the y coordinates in the current x_interval and store values in the output_channel
            for y_cell in range(3):

                # if len(sorted_y) is 0:
                if len(x_sorted_by_y) is 0:
                    discretized_pointcloud[0, x_cell, y_cell] = 0
                    discretized_pointcloud[1, x_cell, y_cell] = 'NaN'
                    discretized_pointcloud[2, x_cell, y_cell] = 'NaN'
                    discretized_pointcloud[3, x_cell, y_cell] = 'NaN'
                else:
                    y_interval = np.asarray(x_sorted_by_y[list(map(lambda x: spatial_resolution * y_cell  < x <= (y_cell + 1) * spatial_resolution, x_sorted_by_y[:, 1]))])

                    # if there are detections save these in right channel
                    if np.shape(y_interval)[0] is not 0:
                        discretized_pointcloud[0, x_cell, y_cell] = np.shape(y_interval)[0]
                        discretized_pointcloud[1, x_cell, y_cell] = np.mean(y_interval[:, 2])
                        discretized_pointcloud[2, x_cell, y_cell] = np.max(y_interval[:, 2])
                        discretized_pointcloud[3, x_cell, y_cell] = np.min(y_interval[:, 2])

                    # if there are not any detections
                    else:
                        discretized_pointcloud[0, x_cell, y_cell] = 0
                        discretized_pointcloud[1, x_cell, y_cell] = 'NaN'
                        discretized_pointcloud[2, x_cell, y_cell] = 'NaN'
                        discretized_pointcloud[3, x_cell, y_cell] = 'NaN'

    print(discretized_pointcloud)
    return discretized_pointcloud

discretized_pointcloud = discretize_pointcloud(point_cloud, 10)

'''
# sort by x

array_to_fill = np.zeros([4, 600, 600])

if len(point_cloud) is 0:

    array_to_fill[0, :, :] = 0
    array_to_fill[1, :, :] = 'NaN'
    array_to_fill[2, :, :] = 'NaN'
    array_to_fill[3, :, :] = 'NaN'

else:

    sorted_x = np.asarray(sorted(point_cloud, key=lambda row: row[0]))

    for i in range(600):
        x_list = list(map(lambda x: grid_size*i < x <= (i+1)*grid_size, sorted_x[:, 0]))
        #print('x_list', x_list)
        x_list = sorted_x[x_list]

        print('sort x', x_list)
        # sort by y
        sorted_y = np.asarray(sorted(x_list, key=lambda row: row[1]))
        #print('sorted_y', sorted_y)
        #print(len(sorted_y))

        for j in range(600):

            if len(sorted_y) is 0:
                array_to_fill[0, i, j] = 0
                array_to_fill[1, i, j] = 'NaN'
                array_to_fill[2, i, j] = 'NaN'
                array_to_fill[3, i, j] = 'NaN'
            else:
                y_list = np.asarray(sorted_y[list(map(lambda x: grid_size*j < x <= (j+1)*grid_size, sorted_y[:, 1]))])
                #print('y_list', y_list)

                if np.shape(y_list)[0] is not 0:
                    #print('enter first if')

                    #print('y_list', y_list)
                    #print('shape', np.shape(y_list))
                    array_to_fill[0, i, j] = np.shape(y_list)[0]
                    array_to_fill[1, i, j] = np.mean(y_list[:, 2])
                    array_to_fill[2, i, j] = np.max(y_list[:, 2])
                    array_to_fill[3, i, j] = np.min(y_list[:, 2])
                    #print(array_to_fill)

                else:
                    #print('enter else')
                    #print('y_list', y_list)
                    #print('shape ', np.shape(y_list))
                    array_to_fill[0, i, j] = 0
                    array_to_fill[1, i, j] = 'NaN'
                    array_to_fill[2, i, j] = 'NaN'
                    array_to_fill[3, i, j] = 'NaN'



'''


