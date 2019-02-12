import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from pathlib import Path


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


def rotate_pointcloud(pointcloud, global_coordinates):
    '''
    Rotate pointcloud (before trimming it).
    :param pointcloud: input raw point cloud shape (N, 3)
    :param global_coordinates: global coordinates for LiDAR, which cintains yaw angle
    :return: rotated_pointcloud: rotated pointcloud, but coordinates are stil relative to LiDAR (not global)
    '''

    number_of_points = len(pointcloud)  # number of points in the pointcloud

    yaw = np.deg2rad(global_coordinates[2])  # convert yaw in degrees to radians
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))) # Rotation matrix

    rotated_pointcloud = Rz @ np.transpose(pointcloud) # rotate each vector with coordinates, transpose to get dimensions correctly
    rotated_pointcloud = np.transpose(np.reshape(rotated_pointcloud, (3, number_of_points)))  # reshape and transpose back

    return rotated_pointcloud


def trim_pointcloud(point_cloud, range=15, roof=10, floor=-3): # the hard coded numbers are not absolute.
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
    points_in_range = np.max(np.absolute(point_cloud), axis=1) <= range  # this takes care of both x, y,and z
    point_cloud = point_cloud[points_in_range]

    # Remove points that are more then roof meters above and floor meters below LiDAR

    z_coordinates = point_cloud[:, -1]
    coordinate_rows = list(map(lambda x: floor <= x <= roof, z_coordinates))
    point_cloud = point_cloud[coordinate_rows]

    return point_cloud


def discretize_pointcloud(trimmed_point_cloud, spatial_resolution=0.05):
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
        #discretized_pointcloud[1, :, :] = 'NaN'
        #discretized_pointcloud[2, :, :] = 'NaN'
        #discretized_pointcloud[3, :, :] = 'NaN'

    else:

        # sort the point cloud by x in increasing order
        x_sorted_point_cloud = np.asarray(sorted(trimmed_point_cloud, key=lambda row: row[0]))

        for x_cell in range(600):

            # get the x-values in the spatial resolution interval
            lower_bound = spatial_resolution * x_cell - 15
            upper_bound = (x_cell + 1) * spatial_resolution - 15
            x_interval = list(map(lambda x: lower_bound < x <= upper_bound, x_sorted_point_cloud[:, 0]))


            x_interval = x_sorted_point_cloud[x_interval]

            # sort the x-interval by increasing y
            x_sorted_by_y = np.asarray(sorted(x_interval, key=lambda row: row[1]))

            # loop through the y coordinates in the current x_interval and store values in the output_channel
            for y_cell in range(600):

                # if len(sorted_y) is 0:
                if len(x_sorted_by_y) is 0:
                    discretized_pointcloud[0, x_cell, y_cell] = 0
                    #discretized_pointcloud[1, x_cell, y_cell] = 'NaN'
                    #discretized_pointcloud[2, x_cell, y_cell] = 'NaN'
                    #discretized_pointcloud[3, x_cell, y_cell] = 'NaN'
                else:
                    lower_bound = spatial_resolution * y_cell - 15
                    upper_bound = (y_cell + 1) * spatial_resolution - 15
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
                        #discretized_pointcloud[1, x_cell, y_cell] = 'NaN'
                        #discretized_pointcloud[2, x_cell, y_cell] = 'NaN'
                        #discretized_pointcloud[3, x_cell, y_cell] = 'NaN'

    # we should normalise the intensity
    # we should convert all nan-values to something else, either here or declare everything as zeros in the beginning


    return discretized_pointcloud


def array_to_png(discretized_pointcloud):
    '''
    Create a png-image of a discretized pointcloud. Create one image per layer. This is mostly for visualizing purposes.
    :param
        discretized_pointcloud:
    :return:
    '''

    # Ask what the png files should be named and create a folder where to save them
    input_folder_name = input('Type name of folder to store png files in: "png_date_number" :')

    # create a folder name
    folder_name = '/_out_' + input_folder_name

    # creates folder to store the png files
    current_path = os.getcwd()
    folder_path = current_path + folder_name

    try:
        os.mkdir(folder_path)
    except OSError:
        print('Failed to create new directory.')
    else:
        print('Successfully created new directory with path: ', folder_path)

    # NORMALIZE THE BEV IMAGE
    for channel in range(np.shape(discretized_pointcloud)[0]):
        max_value = np.max(discretized_pointcloud[channel, :, :])
        # print('Max max_value: ', max_value)
        scale = 255/max_value
        discretized_pointcloud[channel, :, :] = discretized_pointcloud[channel, :, :] * scale
        print('Largest pixel value (should be 255) : ', np.max(discretized_pointcloud[channel, :, :]))
        # create the png_path
        png_path = folder_path + '/_channel' + str(channel)+'.png'

    # Save images
        img = Image.fromarray(discretized_pointcloud[channel, :, :])
        new_img = img.convert("L")
        new_img.save(png_path)


def random_rigid_transformation(bound_translation_meter, bound_rotation_degrees):
    '''
    This functions return an array with values for translation and rotation from ground truth. The values are drawn from
    a distribution given by the user. This rotation/translation should be used to transform the LiDAR sweep to create a
    training sample. Yields equally probable values around 0, both negative and positive up to the given bound.
    Should be used before discretizing the sweep.
    :param bound_translation_meter: scalar, the largest translation value that is acceptable in meters
    :param bound_rotation_degrees: scalar, the largest rotation value that is acceptable in degrees
    :return: rigid_transformation: an array with 3 elements [x, y, angle_degrees]
    '''

    translation = np.random.uniform(-bound_translation_meter, bound_translation_meter, 2)
    rotation = np.random.uniform(-bound_rotation_degrees, bound_rotation_degrees, 1)

    rigid_transformation = np.concatenate((translation, rotation))

    return rigid_transformation
