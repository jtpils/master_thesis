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


'''
def training_sample_rotation_translation(pointcloud, rigid_transformation):
    
    Rotate and translate a pointcloud according to the random rigid transform. Use this when creating fake training samples.
    :param pointcloud: a lidar sweep that is to be rotated/translated in order to create training sample.

    Do this BEFORE trimming the sweep!

    :param rigid_transformation: get a random transformation with function random_rigid_transformation
    :return: training_pointcloud
    
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
    '''


def training_sample_rotation(pointcloud, rotation_angle):
    '''
    Rotate and translate a pointcloud according to the random rigid transform. Use this when creating fake training samples.
    :param pointcloud: a lidar sweep that is to be rotated/translated in order to create training sample.

    Do this BEFORE trimming the sweep!

    :param rotation_angle: get a random transformation with function random_rigid_transformation
    :return: training_pointcloud
    '''
    number_of_points = len(pointcloud)  # number of points in the pointcloud
    rotation = np.deg2rad(rotation_angle)

    # rotate:
    c, s = np.cos(rotation), np.sin(rotation)
    Rz = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))) # Rotation matrix

    rotated_pointcloud = Rz @ np.transpose(pointcloud) # rotate each vector with coordinates, transpose to get dimensions correctly
    rotated_pointcloud = np.transpose(np.reshape(rotated_pointcloud, (3, number_of_points)))  # reshape and transpose back

    return rotated_pointcloud

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


def rounding(n, r=0.05): # do not forget to change this to 0.1!! it should be the same as the spatial resolution
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


def discretize_pc_fast(pc, trim_range=15, spatial_resolution=0.1, array_size=300):
    # lidar must be in origin
    x_grids = np.floor((pc[:,0]+trim_range)/spatial_resolution).astype(int)
    y_grids = np.floor((pc[:,1]+trim_range)/spatial_resolution).astype(int)

    # not the best way to handle detections outside of array, would be better to make the array bigger like 2 cells larger, do as below, and then crop our image from it.
    pc_image = np.zeros((1,array_size+1,array_size+1))
    for i in np.arange(len(pc)):
        pc_image[0, x_grids[i],y_grids[i]] = pc_image[0, x_grids[i], y_grids[i]] + 1

    pc_image = pc_image[:,:-1,:-1]
    return pc_image
