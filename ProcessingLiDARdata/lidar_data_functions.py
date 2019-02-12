import numpy as np


def load_data(path_to_ply, path_to_csv):
    '''
    Load the data from a .ply-file, toghether with the global coordinates for this fram from a csv-file.
    :param
        path_to_ply: string with path to .ply-file
        path_to_csv: string with path to csv-file
    :return:
        pointcloud: nd-array with xyz-coordinates for all detections in the .ply-file. Shape (N,3).
        global_lidar_coordinates: global xyz-coordinates and yaw in degrees for LiDAR position, [x y z yaw]

    TO DO IN THE FUTURE:
    Maybe its unnecessary to read the csv every time. Perhaps do this part once outside the function only once for every ply-file.
    '''

    # load pointcloud
    point_cloud = np.loadtxt(path_to_ply, skiprows=7)

    # extract frame_number from filename
    file_name = path_to_ply.split('/')[-1] # keep the last part of the path, i.e. the file name
    frame_number = int(file_name[:-4]) # remove the part '.ply' and convert to int

    # load csv-file with global coordinates
    global_coordinates = np.loadtxt(path_to_csv, skiprows=1, delimiter=',')

    # extract information from csv at given frame_number
    row = np.where(global_coordinates==frame_number)[0]  # returns which row the frame number is located on
    global_lidar_coordinates = global_coordinates[row, 1:5]

    return point_cloud, global_lidar_coordinates


def rotate_translate_pointcloud(pointcloud, global_coordinates):
    '''
    Rotate and translate pointcloud before trimming it.
    :param pointcloud:
    :param global_coordinates:
    :return: pointcloud
    '''

    return pointcloud


def trim_pointcloud(point_cloud, range=15, roof=10, floor=3): # the hard coded numbers are not absolute.
    '''
    Trim pointcloud to a range given by range_of_interest. Trim detections that are more than (roof) meters above LiDAR,
    and more than (floor) meters below LiDAR. After this, remove z.
    :param
        pointcloud: ndarray size (N, 3)
        region_of_interest: range
        roof: meter
        floor: meter
    :return:
        2D_pointcloud: nd-array with xy-coordinates, with shape (N, 2)
    '''

    # remove points outside the range of interest

    points_in_range = list(map(lambda x: -range <= x <= range, np.max(np.absolute(point_cloud), axis=1)))
    # points_in_range = np.max(np.absolute(point_cloud), axis=1) < range
    point_cloud = point_cloud[points_in_range]

    # Remove points that are more then roof meters above and floor meters below LiDAR

    z_coordinates = point_cloud[:, -1]
    coordinate_rows = list(map(lambda x: floor <= x <= roof, z_coordinates))

    point_cloud = point_cloud[coordinate_rows]

    return point_cloud


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

    discretized_pointcloud = np.zeros([4, 600, 600])

    if len(trimmed_point_cloud) is 0:

        discretized_pointcloud[0, :, :] = 0
        discretized_pointcloud[1, :, :] = 'NaN'
        discretized_pointcloud[2, :, :] = 'NaN'
        discretized_pointcloud[3, :, :] = 'NaN'

    else:

        # sort the point cloud by x in increasing order
        x_sorted_point_cloud = np.asarray(sorted(trimmed_point_cloud, key=lambda row: row[0]))

        for x_cell in range(600):

            # get the x-values in the spatial resolution interval
            x_interval = list(map(lambda x: (spatial_resolution * x_cell) < x <= (x_cell + 1) * spatial_resolution, x_sorted_point_cloud[:, 0]))
            x_interval = x_sorted_point_cloud[x_interval]

            # sort the x-interval by increasing y
            x_sorted_by_y = np.asarray(sorted(x_interval, key=lambda row: row[1]))

            # loop through the y coordinates in the current x_interval and store values in the output_channel
            for y_cell in range(600):

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

    return discretized_pointcloud


def array_to_png(discretized_pointcloud):
    '''
    Create a png-image of a discretized pointcloud. Create one image per layer. This is mostly for visualizing purposes. (?)
    :param
        discretized_pointcloud:
    :return:
    '''


