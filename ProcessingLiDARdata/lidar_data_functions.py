import numpy as np
from PIL import Image
import os
import sys


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
    point_cloud[:, 1] = - point_cloud[:, 1]
    point_cloud[:, -1] = - point_cloud[:, -1]

    # check if ply file is empty. Stop execution if it is.
    if np.shape(point_cloud)[0] == 0:
        print("ply file with point cloud is empty")
        sys.exit()

    # extract frame_number from filename
    file_name = path_to_ply.split('/')[-1]  # keep the last part of the path, i.e. the file name
    frame_number = int(file_name[:-4])  # remove the part '.ply' and convert to int
    #print('frame number', frame_number)
    # load csv-file with global coordinates
    global_coordinates = np.loadtxt(path_to_csv, skiprows=1, delimiter=',')

    # extract information from csv at given frame_number
    row = np.where(global_coordinates == frame_number)[0]  # returns which row the frame number is located on
    #print('row where to find frame number', row)
    global_lidar_coordinates = global_coordinates[row, 1:5]
    global_lidar_coordinates[0][3] = global_lidar_coordinates[0][3] + 90

    return point_cloud, global_lidar_coordinates #, frame_number


def rotate_pointcloud_to_global(pointcloud, global_coordinates):
    '''
    Rotate pointcloud (before trimming it).
    :param pointcloud: input raw point cloud shape (N, 3)
    :param global_coordinates: global coordinates for LiDAR, which contains yaw angle
    :return: rotated_pointcloud: rotated pointcloud, but coordinates are stil relative to LiDAR (not global)
    '''

    number_of_points = len(pointcloud)  # number of points in the pointcloud

    yaw = np.deg2rad(global_coordinates[0, 3])  # convert yaw in degrees to radians
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))) # Rotation matrix

    rotated_pointcloud = Rz @ np.transpose(pointcloud) # rotate each vector with coordinates, transpose to get dimensions correctly
    rotated_pointcloud = np.transpose(np.reshape(rotated_pointcloud, (3, number_of_points)))  # reshape and transpose back

    return rotated_pointcloud


def translate_pointcloud_to_global(pointcloud, global_coordinates):
    '''
    Translates all the points in the pointcloud to the global coordinates. This should be done after rotating and then
    trimming the pointcloud.
    :param pointcloud:
    :param global_coordinates:
    :return: global_pointcloud
    '''

    global_pointcloud = pointcloud + global_coordinates[0, :3]  #only add xyz (not yaw)

    return global_pointcloud


def trim_pointcloud(point_cloud, range=15, roof=10, floor=-3): # the hard coded numbers are not absolute.
    '''
    Trim pointcloud to a range given by range_of_interest. Trim detections that are more than (roof) meters above LiDAR,
    and more than (floor) meters below LiDAR. After this, remove z.

    The floor and ground parameter represents the interval of points that should be kept. This means sending in floor=0
    and roof=0 only the detectons from the ground will be kept.

    :param
        pointcloud: ndarray size (N, 3)
        region_of_interest: range
        roof: meter above ground
        floor: meter above ground
    :return:
        2D_pointcloud: nd-array with xy-coordinates, with shape (N, 2)
    '''

    # remove points outside the range of interest
    points_in_range = np.max(np.absolute(point_cloud), axis=1) <= range  # this takes care of both x, y,and z
    point_cloud = point_cloud[points_in_range]

    z_coordinates = point_cloud[:, -1]
    # Remove points that are more then loor and roof meters above ground coordinate
    ground_coordinate = min(z_coordinates)

    floor = ground_coordinate + floor
    roof =  ground_coordinate + roof

    coordinate_rows = list(map(lambda x: floor <= x <= roof, z_coordinates))

    point_cloud = point_cloud[coordinate_rows]

    return point_cloud

def discretize_pointcloud(trimmed_point_cloud, array_size=600, trim_range=15, spatial_resolution=0.05, padding=False, pad_size=150):
    '''
    Discretize into a grid structure with different channels.
    Create layers:
    1 layer with number of detections (normalise at the end), 1 layer with mean height, max height, min height. other layers?
    :param
        trimmed_point_cloud: trimmed point cloud
        array_size: number of cells in each channel in the discretized point cloud (default=600)
        trim_range: move coordinate system such that it matches the range of the trimmed point cloud (default=15)
        spatial_resolution: size of grid cell in m (default=0.05)
        padding: True if padding should be performed False otherwise (default=false)
        pad_size: size of padding (default=150)
    :return:
        discretized_pointcloud: 3d-array with multiple "layers"
        layer 0 = number of detections
        layer 1 = mean evaluation
        layer 2 = maximum evaluation
        layer 3 = minimum evaluation
    '''
    array_size = int(array_size)
    discretized_pointcloud = np.zeros([4, array_size, array_size])

    if len(trimmed_point_cloud) is 0:

        discretized_pointcloud[0, :, :] = 0

    else:

        # sort the point cloud by x in increasing order
        x_sorted_point_cloud = np.asarray(sorted(trimmed_point_cloud, key=lambda row: row[0]))

        for x_cell in range(array_size):

            # get the x-values in the spatial resolution interval
            lower_bound = spatial_resolution * x_cell - trim_range
            upper_bound = (x_cell + 1) * spatial_resolution - trim_range
            x_interval = list(map(lambda x: lower_bound < x <= upper_bound, x_sorted_point_cloud[:, 0]))

            x_interval = x_sorted_point_cloud[x_interval]

            # sort the x-interval by increasing y
            x_sorted_by_y = np.asarray(sorted(x_interval, key=lambda row: row[1]))

            # loop through the y coordinates in the current x_interval and store values in the output_channel
            for y_cell in range(array_size):

                # if len(sorted_y) is 0:
                if len(x_sorted_by_y) is 0:
                    discretized_pointcloud[0, x_cell, y_cell] = 0
                else:
                    lower_bound = spatial_resolution * y_cell - trim_range
                    upper_bound = (y_cell + 1) * spatial_resolution - trim_range
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

    # pad the discretized point cloud
    if padding:
        discretized_pointcloud = np.pad(discretized_pointcloud, [(0, 0), (pad_size, pad_size), (pad_size, pad_size)], mode='constant')
    # Normalize the channels. The values should be between 0 and 1
    for channel in range(np.shape(discretized_pointcloud)[0]):
        max_value = np.max(discretized_pointcloud[channel, :, :])
        # print('Max max_value inarray_to_png: ', max_value)

        # avoid division with 0
        if max_value == 0:
            max_value = 1

        scale = 1/max_value
        discretized_pointcloud[channel, :, :] = discretized_pointcloud[channel, :, :] * scale
        # print('Largest pixel value (should be 1) : ', np.max(discretized_pointcloud[channel, :, :]))

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
        print('Max max_value inarray_to_png: ', max_value)

        # avoid division with 0
        if max_value == 0:
            max_value = 1

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


def training_sample_rotation_translation(pointcloud, rigid_transformation):
    '''
    Rotate and translate a pointcloud according to the random rigid transform. Use this when creating training samples.
    :param pointcloud: a lidar sweep that is to be rotated/translated in order to create training sample.

    Do this BEFORE trimming the sweep!

    :param rigid_transformation: get a random transformation with function random_rigid_transformation
    :return: training_pointcloud
    '''
    number_of_points = len(pointcloud)  # number of points in the pointcloud
    translation = np.append(rigid_transformation[:2], 0) # add a zero for z, since we do not want to translate the height
    rotation = np.deg2rad(rigid_transformation[-1])

    # rotate:
    c, s = np.cos(rotation), np.sin(rotation)
    Rz = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))) # Rotation matrix

    rotated_pointcloud = Rz @ np.transpose(pointcloud) # rotate each vector with coordinates, transpose to get dimensions correctly
    rotated_pointcloud = np.transpose(np.reshape(rotated_pointcloud, (3, number_of_points)))  # reshape and transpose back

    # translate:
    training_pointcloud = rotated_pointcloud + translation # add translation to every coordinate vector

    return training_pointcloud
