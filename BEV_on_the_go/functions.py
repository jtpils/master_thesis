import numpy as np
from PIL import Image
import os
import sys
import pandas as pd
import math
from matplotlib import pyplot as plt
from tqdm import tqdm

def load_data(path_to_ply, path_to_csv):
    # load pointcloud
    point_cloud = pd.read_csv(path_to_ply, delimiter=' ', skiprows=7, header=None, names=('x','y','z'))
    point_cloud = point_cloud.values
    point_cloud[:, -1] = - point_cloud[:, -1]  # z = -z, get the correct coordinate system

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


def trim_point_cloud_range(point_cloud, trim_range=15):
    # keep points inside the range of interest
    points_in_range = np.max(np.absolute(point_cloud), axis=1) <= trim_range  # this takes care of both x, y and z
    point_cloud = point_cloud[points_in_range]

    return point_cloud


def trim_point_cloud_vehicle_ground(point_cloud, remove_vehicle=True, remove_ground=True):
    # keep points inside the range of interest

    if remove_ground:
        z_coordinates = point_cloud[:, -1]
        ground_coordinate = min(z_coordinates)
        floor = ground_coordinate + 0.05
        coordinate_rows = list(map(lambda x: floor <= x, z_coordinates))
        point_cloud = point_cloud[coordinate_rows]

    if remove_vehicle:
        points_in_range = np.max(np.absolute(point_cloud), axis=1) >= 2.3  # this takes care of both x, y and z
        point_cloud = point_cloud[points_in_range]

    return point_cloud


def discretize_point_cloud(point_cloud, trim_range=15, spatial_resolution=0.1, image_size=300):
    # obs, the point cloud has lidar in origin

    x_grids = np.floor((point_cloud[:,0] + trim_range)/spatial_resolution).astype(int)
    y_grids = np.floor((point_cloud[:,1] + trim_range)/spatial_resolution).astype(int)

    image = np.zeros((1,image_size+1, image_size+1))
    for i in np.arange(len(point_cloud)):
        image[:,x_grids[i],y_grids[i]] = image[:,x_grids[i],y_grids[i]] + 1

    image = image[:,:300,:300]
    return image


def random_rigid_transformation(bound_translation_meter, bound_rotation_degrees):
    translation = np.random.uniform(-bound_translation_meter, bound_translation_meter, 2)
    rotation = np.random.uniform(-bound_rotation_degrees, bound_rotation_degrees, 1)

    rigid_transformation = np.concatenate((translation, rotation))

    return rigid_transformation


def rotate_point_cloud(point_cloud, rotation_angle):
    rotation = np.deg2rad(rotation_angle)

    # rotate:
    c, s = np.cos(rotation), np.sin(rotation)
    Rz = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))) # Rotation matrix

    rotated_point_cloud = Rz @ np.transpose(point_cloud) # rotate each vector with coordinates, transpose to get dimensions correctly
    rotated_point_cloud = np.transpose(rotated_point_cloud)

    return rotated_point_cloud

def translate_point_cloud(point_cloud, translation):
    translation = np.array((translation[0], translation[1], 0))
    point_cloud = point_cloud + translation

    return point_cloud


def normalize_sample(sample):
    for layer in (0, 1):
        max_value = np.max(sample[layer, :, :])

        # avoid division with 0
        if max_value == 0:
            max_value = 1

        scale = 1 / max_value
        sample[layer, :, :] = sample[layer, :, :] * scale

    return sample


def visualize_detections(discretized_point_cloud, fig_num=1):
    detection_layer = discretized_point_cloud[0, :, :]
    detection_layer[detection_layer > 0] = 255

    plt.figure(fig_num)
    plt.imshow(detection_layer, cmap='gray')
    #plt.show()


