import numpy as np
from PIL import Image
import os
import sys
import pandas as pd
import math
from matplotlib import pyplot as plt
from tqdm import tqdm


def load_data(path_to_ply, path_to_csv):
    '''
    Load the data from a .ply-file, toghether with the global coordinates for this from from a csv-file.
    :param
        path_to_ply: string with path to .ply-file
        path_to_csv: string with path to .csv-file
    :return:
        pointcloud: ndarray with xyz-coordinates for all detections in the .ply-file. Detections are relative to LiDAR position.
        global_lidar_coordinates: global xyz-coordinates and yaw in degrees for LiDAR position, [x y z yaw]
    '''

    # load pointcloud
    point_cloud = pd.read_csv(path_to_ply, delimiter=' ', skiprows=7, header=None, names=('x','y','z'))
    point_cloud = point_cloud.values
    point_cloud[:, -1] = - point_cloud[:, -1]  # z = -z, get the correct coordinate system

    # check if ply file is empty. Stop execution if it is.
    if np.shape(point_cloud)[0] == 0:
        print("ply file with point cloud is empty")
        #sys.exit()

    # extract frame_number from filename
    file_name = path_to_ply.split('/')[-1]  # keep the last part of the path, i.e. the file name
    frame_number = int(file_name[:-4])  # remove the part '.ply' and convert to int

    # load csv-file with global coordinates
    global_coordinates = pd.read_csv(path_to_csv)
    global_coordinates = global_coordinates.values

    # extract information from csv at given frame_number
    row = np.where(global_coordinates == frame_number)[0]  # returns which row the frame number is located on
    global_lidar_coordinates = global_coordinates[row, 1:5]  # save that row, columns [x y z yaw]

    # Fix yaw angle:
    if global_lidar_coordinates[0][3] < 0:
        global_lidar_coordinates[0][3] = global_lidar_coordinates[0][3] + 360  # change angles to interval [0,360]
    global_lidar_coordinates[0][3] = global_lidar_coordinates[0][3] + 90  # add 90 to get the correct coordinate system

    global_lidar_coordinates[0][1] = -global_lidar_coordinates[0][1]  # y = -y, get the correct coordinate system

    return point_cloud, global_lidar_coordinates


def rotate_pointcloud_to_global(pointcloud, global_coordinates):
    '''
    Rotate pointcloud (before trimming it).
    :param pointcloud: input raw point cloud shape (N, 3)
    :param global_coordinates: global coordinates for LiDAR, which contains yaw angle
    :return: rotated_pointcloud: rotated pointcloud, but coordinates are stil relative to LiDAR (not global)
    '''

    yaw = np.deg2rad(global_coordinates[0, 3]) # convert yaw in degrees to radians
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))  # Rotation matrix
    rotated_pointcloud = Rz @ np.transpose(pointcloud)  # rotate each vector with coordinates, transpose to get dimensions correctly
    rotated_pointcloud = np.transpose(rotated_pointcloud)

    rotated_pointcloud[:, 1] = -rotated_pointcloud[:,1]  # y = -y, get the correct coordinate system

    return rotated_pointcloud


def translate_pointcloud_to_global(pointcloud, global_coordinates):
    '''
    Translates all the points in the pointcloud to the global coordinates. This should be done after rotating and then
    trimming the pointcloud.
    :param pointcloud:
    :param global_coordinates:
    :return: global_pointcloud
    '''

    global_pointcloud = pointcloud + global_coordinates[0, :3]  # only add xyz (not yaw)

    return global_pointcloud


def trim_pointcloud(point_cloud, range=15, roof=10, floor=0):  # the hard coded numbers are not absolute.
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
    # Remove points that are more then floor and roof meters above ground coordinate
    ground_coordinate = min(z_coordinates)

    floor = ground_coordinate + floor
    roof = ground_coordinate + roof

    coordinate_rows = list(map(lambda x: floor <= x <= roof, z_coordinates))

    point_cloud = point_cloud[coordinate_rows]

    return point_cloud


def discretize_pointcloud(trimmed_point_cloud, array_size=300, trim_range=15, spatial_resolution=0.1, padding=False, pad_size=150):
    '''
    Discretize into a grid structure with different channels.
    Create layers:
    1 layer with number of detections (normalise at the end), 1 layer with mean height, max height, min height. other layers?
    :param
        trimmed_point_cloud: trimmed point cloud
        array_size: number of cells in each channel in the discretized point cloud (default=600)
        trim_range: move coordinate system such that it matches the range of the trimmed point cloud (default=15)
        spatial_resolution: size of grid cell in m (default=0.05)
        padding: True if padding should be performed False otherwise (default=True)
        pad_size: size of padding (default=150)
    :return:
        discretized_pointcloud: 3d-array with multiple "layers"
        layer 0 = number of detections
        layer 1 = mean evaluation
        layer 2 = maximum evaluation
        layer 3 = minimum evaluation
    '''

    array_size = int(array_size)
    discretized_pointcloud = np.zeros([1, array_size, array_size])

    if len(trimmed_point_cloud) is 0:

        discretized_pointcloud[0, :, :] = 0

    else:

        # sort the point cloud by x in increasing order
        x_sorted_point_cloud = np.asarray(sorted(trimmed_point_cloud, key=lambda row: row[0]))

        for x_cell in tqdm(range(array_size)):

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
                        #discretized_pointcloud[1, x_cell, y_cell] = np.mean(y_interval[:, 2])

                    # if there are no detections not necessary already initialized to zero
                    #else:
                    #    discretized_pointcloud[0, x_cell, y_cell] = 0

    # pad the discretized point cloud
    if padding:
        discretized_pointcloud = np.pad(discretized_pointcloud, [(0, 0), (pad_size, pad_size), (pad_size, pad_size)], mode='constant')


    # THIS IS DONE IN A SEPARAT FUNCTION INSTEAD
    # Normalize the channels. The values should be between 0 and 1
    '''for channel in range(np.shape(discretized_pointcloud)[0]):
        max_value = np.max(discretized_pointcloud[channel, :, :])

        # avoid division with 0
        if max_value == 0:
            max_value = 1

        scale = 1/max_value
        discretized_pointcloud[channel, :, :] = discretized_pointcloud[channel, :, :] * scale'''

    return discretized_pointcloud


def discretize_pointcloud_map(map_point_cloud, min_max_coordinates, spatial_resolution=0.05):
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
    number_x_cells = int(np.ceil((x_max - x_min) / spatial_resolution))  # should there be a +1 or -1 or something like that? Now Sabina has set 10 of some reason she can't explain.
    number_y_cells = int(np.ceil((y_max - y_min) / spatial_resolution))  # should there be a +1 or -1 or something like that?
    # print('number of cells', number_x_cells)

    discretized_pointcloud = np.zeros([4, number_x_cells, number_y_cells])

    # sort the point cloud by x in increasing order
    x_sorted_point_cloud = np.asarray(sorted(map_point_cloud, key=lambda row: row[0]))
    i = 0
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
        i += 1

        if i%100 == 0:
            print('Number of x cells checked:', i, 'of:', number_x_cells)

    return discretized_pointcloud



def array_to_png(discretized_pointcloud, max_min_values):
    '''
    Create a png-image of a discretized pointcloud. Create one image per layer. This is mostly for visualizing purposes.
    Also saves the map matrix and the max min values.
    :param
        discretized_pointcloud: The discretized point cloud map matrix
        max_min_values: The max and min values of the point cloud map.
    :return:
    '''

    # Ask what the png files should be named and create a folder where to save them
    input_folder_name = input('Type name of folder to store png files in: "map_date_number" :')

    # create a folder name
    folder_name = input_folder_name

    # creates folder to store the png files
    current_path = os.getcwd()
    folder_path = os.path.join(current_path,folder_name)
    folder_path_png = folder_path + '/map_png/'
    try:
        os.mkdir(folder_path)
        os.mkdir(folder_path_png)
    except OSError:
        print('Failed to create new directory.')
    else:
        print('Successfully created new directory with path: ', folder_path, 'and', folder_path_png)

    discretized_pointcloud_BEV = discretized_pointcloud  # Save map in new variable to be scaled
    # NORMALIZE THE BEV IMAGE
    for channel in range(np.shape(discretized_pointcloud_BEV)[0]):
        max_value = np.max(discretized_pointcloud_BEV[channel, :, :])
        print('Max max_value inarray_to_png: ', max_value)

        # avoid division with 0
        if max_value == 0:
            max_value = 1

        scale = 255/max_value
        discretized_pointcloud_BEV[channel, :, :] = discretized_pointcloud_BEV[channel, :, :] * scale
        print('Largest pixel value (should be 255) : ', np.max(discretized_pointcloud_BEV[channel, :, :]))
        # create the png_path
        png_path = folder_path_png + 'channel_' + str(channel)+'.png'

    # Save images
        img = Image.fromarray(discretized_pointcloud_BEV[channel, :, :])
        new_img = img.convert("L")
        new_img.rotate(180).save(png_path)

    # Save the map array and the max and min values of the map in the same folder as the BEV image
    np.save(os.path.join(folder_path, 'map.npy'), discretized_pointcloud)
    np.save(os.path.join(folder_path, 'max_min.npy'), max_min_values)


def random_rigid_transformation(bound_translation_meter, bound_rotation_degrees):
    '''
    This functions return an array with values for translation and rotation from ground truth. The values are drawn from
    a uniform distribution given by the user. This rotation/translation should be used to transform the LiDAR sweep to create a
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
    Rotate and translate a pointcloud according to the random rigid transform. Use this when creating fake training samples.
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


def normalize_sample(sample):
    '''
    Normalize the first and 4th layer wrt (channel 0 and 4) wrt the highest number of detections. Each layer is processed
    individually. Channel 1,2,3,5,6,7 is normalized wrt the max height in the whole sample. Returns a normalized sample
    :sample: the concatenated sweep and cut out, representing a sample.
    :return: normalized_sample
    '''

    # Normalize number of detections:
    for layer in (0, 1):#, 2):  # normalize number of detections in layer 0 and layer 2
        max_value = np.max(sample[layer, :, :])

        # avoid division with 0
        if max_value == 0:
            max_value = 1

        scale = 1 / max_value
        sample[layer, :, :] = sample[layer, :, :] * scale

    # Normalize height wrt to max value of both map and sweep:
    '''max_height = 0
    for layer in (1, 3):  # max height in sweep and cutout
        max_height_temp = np.max(sample[layer, :, :])
        if max_height_temp > max_height:
            max_height = max_height_temp

    # avoid division with 0
    if max_height == 0:
        max_height = 1  # nothing happens when we normalize

    scale = 1 / max_height
    for layer in (1, 3):  # height channels normalized wrt to the same max value
        sample[layer, :, :] = sample[layer, :, :] * scale

        normalized_sample = sample'''

    return sample


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

    cell_check_x = rounding(x_min, spatial_resolution) #- pad_size_low*spatial_resolution
    # print('start_cell_check', cell_check_x)
    k = pad_size_low
    while cell_check_x < global_coordinate[0]:
        x_cell = k
        k += 1
        cell_check_x += spatial_resolution
    # print('x_cell: ', x_cell)

    # Check the y value and "put" it in the right cell of the map
    # Bounds must be fixed since the map now is bigger and have zeros! Lower bound of y is now (y_min - pad_size_low)
    # and Upper bound of y is (y_max + pad_size_high)

    cell_check_y = rounding(y_min, spatial_resolution)
    # print('start_cell_check', cell_check_y)
    k = pad_size_low
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


def visualize_detections(discretized_point_cloud, layer=0, fig_num=1):
    '''
    takes as input a discretized point cloud (all channels) and visualizes the detections in channel 0. Every cell with
    at least one detection is assigned the value 255. When showing the image, all detections will appear as white pixels.
    Call plt.show() after this function! Can plot multiple figures.
    :param discretized_point_cloud: pointcloud with 4 channels
           layer = the layer we want to see. Often 0 or 4
           fig_num: figure number
    :return:
    '''
    detection_layer = discretized_point_cloud[layer, :, :]
    detection_layer[detection_layer > 0] = 255

    plt.figure(fig_num)
    plt.imshow(detection_layer, cmap='gray')
