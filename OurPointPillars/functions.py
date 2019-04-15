import numpy as np
from PIL import Image
import os
import sys
import pandas as pd
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import random


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
    global_lidar_coordinates = global_lidar_coordinates[0]

    # Fix yaw angle:
    if global_lidar_coordinates[3] < 0:
        global_lidar_coordinates[3] = global_lidar_coordinates[3] + 360  # change angles to interval [0,360]
    global_lidar_coordinates[3] = global_lidar_coordinates[3] + 90  # add 90 to get the correct coordinate system

    # rotate:
    rotation = np.deg2rad(global_lidar_coordinates[3])
    c, s = np.cos(rotation), np.sin(rotation)
    Rz = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))) # Rotation matrix
    point_cloud = Rz @ np.transpose(point_cloud) # rotate each vector with coordinates, transpose to get dimensions correctly
    point_cloud = np.transpose(point_cloud)

    point_cloud[:, 1] = -point_cloud[:,1]  # y = -y

    global_lidar_coordinates[1] = -global_lidar_coordinates[1]  # y = -y, get the correct coordinate system

    #translate
    point_cloud = point_cloud + global_lidar_coordinates[0:3]

    return point_cloud, global_lidar_coordinates


def trim_point_cloud_range(point_cloud, origin, trim_range=15):
    """

    :param point_cloud:
    :param origin: list with x,y from global coordinates
    :param trim_range:
    :return:
    """
    # translate all points to our origin
    translation = np.array((origin[0], origin[1], 0))
    point_cloud = point_cloud - translation

    # keep points inside the range of interest
    points_in_range = np.max(np.absolute(point_cloud), axis=1) <= trim_range  # this takes care of both x, y and z
    point_cloud = point_cloud[points_in_range]

    # translate back to global coordinates
    point_cloud = point_cloud + translation

    return point_cloud


def trim_point_cloud_vehicle_ground(point_cloud, origin, remove_vehicle=True, remove_ground=True):
    # keep points inside the range of interest

    # translate all points to our origin
    translation = np.array((origin[0], origin[1], 0))
    point_cloud = point_cloud - translation

    if remove_ground:
        z_coordinates = point_cloud[:, -1]
        ground_coordinate = min(z_coordinates)
        floor = ground_coordinate + 0.05
        coordinate_rows = list(map(lambda x: floor <= x, z_coordinates))
        point_cloud = point_cloud[coordinate_rows]

    if remove_vehicle:
        points_in_range = np.max(np.absolute(point_cloud), axis=1) >= 2.3  # this takes care of both x, y and z
        point_cloud = point_cloud[points_in_range]

    # translate back to global coordinates
    point_cloud = point_cloud + translation

    return point_cloud


def random_rigid_transformation(bound_translation_meter, bound_rotation_degrees):
    translation = np.random.uniform(-bound_translation_meter, bound_translation_meter, 2)
    rotation = np.random.uniform(-bound_rotation_degrees, bound_rotation_degrees, 1)

    rigid_transformation = np.concatenate((translation, rotation))

    return rigid_transformation


def rotate_point_cloud(point_cloud, origin, rotation_angle, to_global):
    rotation = np.deg2rad(rotation_angle)

    translation = np.array((origin[0], origin[1], 0))
    point_cloud = point_cloud - translation # translate origin to the middle of the point cloud, else we rotate around something else

    # rotate:
    c, s = np.cos(rotation), np.sin(rotation)
    Rz = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))) # Rotation matrix

    point_cloud = Rz @ np.transpose(point_cloud) # rotate each vector with coordinates, transpose to get dimensions correctly
    point_cloud = np.transpose(point_cloud)

    if to_global:
        point_cloud[:, 1] = -point_cloud[:,1]

    point_cloud = point_cloud + translation

    return point_cloud


def translate_point_cloud(point_cloud, translation):
    translation = np.array((translation[0], translation[1], 0))
    point_cloud = point_cloud + translation

    return point_cloud



def create_pillars(point_cloud, origin, pillar_size=0.5, trim_range=15):
    '''
    Function that creates a dict containing the 8 features for each pillar in a point cloud. Each pillar is grid_size large.
    :param point_cloud: <nd.array> [nx3] nd array containing the x, y, z coordinates of a point cloud
    :param pillar_size: <float> The size of the pillar.
    :return: pillar_dict: <dict> A dict containing the features for each pillar
    '''
    x_edges = np.arange(-trim_range, trim_range, pillar_size)
    y_edges = np.arange(-trim_range, trim_range, pillar_size)

    # translate point cloud such that the center of the cloud is the origin
    point_cloud = point_cloud - np.array((origin[0], origin[1], 0))

    pillar_dict = {}
    pillar_feature_dict = {}
    coordinate_dict = {}
    for row in range(len(point_cloud[:,0])): # TODO: maybe we should sort for x and y as we did before? /A
        # Get which grid the current point belongs to.
        x_grid = np.floor((point_cloud[row,0] + trim_range)/pillar_size).astype(int)
        y_grid = np.floor((point_cloud[row,1] + trim_range)/pillar_size).astype(int)

        cell_name = (x_grid, y_grid)#str(x_grid) + ',' + str(y_grid)

        # If the cell name has been used before concatenate the points and update the value of the key. Else create
        # a new key and add the coordinates of the point.
        if cell_name in pillar_dict.keys():
            cell_value = pillar_dict[cell_name]

            # 1. Get the z value
            z = point_cloud[row, 2]

            # 2. calculate the offset from the pillar x,y center i.e xp and yp.
            x_offset = x_edges[x_grid] + pillar_size/2
            y_offset = y_edges[y_grid] + pillar_size/2

            xp = point_cloud[row, 0] - x_offset
            yp = point_cloud[row, 1] - y_offset

            new_feature = np.array((xp, yp, z))
            cell_value = np.vstack((cell_value, new_feature))

            pillar_dict.update({cell_name: cell_value})

        else:
            z = point_cloud[row, 2]
            #z = z.reshape((np.shape(z)[0], 1))

            # 2. calculate the offset from the pillar x,y center i.e xp and yp.
            x_offset = x_edges[x_grid] + pillar_size/2
            y_offset = y_edges[y_grid] + pillar_size/2

            xp = point_cloud[row, 0] - x_offset
            #xp = xp.reshape((np.shape(xp)[0],1))

            yp = point_cloud[row, 1] - y_offset
            #yp = yp.reshape((np.shape(yp)[0],1))

            new_feature = np.array((xp, yp, z))

            pillar_dict.update({cell_name: new_feature})


        if cell_name not in coordinate_dict.keys():
            coordinate_dict.update({cell_name : point_cloud[row,:]})

    return pillar_dict, coordinate_dict


def get_feature_tensor(pillar_dict, coordinate_dict, max_number_of_pillars=1260, max_number_of_points_per_pillar=900, dimension=3):
    '''
    Function that creates the feature tensor with dimension (D,P,N)
    D = Dimension (3) xp, yp, z
    P = max antal pillars (1260)
    N = maximum points per pillar (900)
    :param pillar_dicts: <dict> Dict containing features for each pillar.
    :param max_number_of_pillars: <int> Max number of pillars in a sample. (default=1260)
    :param max_number_of_points_per_pillar: <int> Max number of points in a sample. (default=900)
    :param dimension: <int> Dimension of features. (default=3)
    :return: feature_tensor
    '''

    # Initialize feature tensor
    feature_tensor = np.zeros((dimension, max_number_of_pillars, max_number_of_points_per_pillar))
    coordinate_tensor = np.zeros((max_number_of_pillars,3))
    # 1. Check how many keys in the dict. If more than max number of pillars pick random max_numer_of_pillars
    number_of_pillars = len(pillar_dict.keys())

    # if number of pillars is more than the maximum allowed. set number of pillars = max_number and sample the key list
    if number_of_pillars > max_number_of_pillars:
        key_list = random.sample(list(pillar_dict), max_number_of_pillars)
    else:
        key_list = list(pillar_dict)

    pillar = 0
    for key in key_list:
        # Get value from dict
        key_value = pillar_dict[key]
        if len(np.shape(key_value))==1:
            key_value = np.expand_dims(pillar_dict[key], axis=0)
        number_of_points = np.shape(key_value)[0]

        if number_of_points > max_number_of_points_per_pillar:

            number_of_points_index = list(range(0,number_of_points-1))
            random_index = random.sample(number_of_points_index, max_number_of_points_per_pillar)

            key_value = key_value[random_index,:]

        feature_tensor[:, pillar, :number_of_points] = key_value.T
        coordinate_tensor[pillar,:] = coordinate_dict[key]
        pillar += 1

    return feature_tensor, coordinate_tensor






# new function /A
def both_pillar_tensor_FAST(point_cloud, origin, pillar_size=0.5, trim_range=15, max_number_of_pillars=1260,
         max_number_of_points_per_pillar=900, dimension=3):
    '''
    Function that creates a dict containing the 8 features for each pillar in a point cloud. Each pillar is grid_size large.
    :param point_cloud: <nd.array> [nx3] nd array containing the x, y, z coordinates of a point cloud
    :param pillar_size: <float> The size of the pillar.
    :return: pillar_feature_dict: <dict> A dict containing the features for each pillar
    '''

    # if we shuffle the points first, can we just fill a pillar in a dict and when we reach max_num-points in that
    # pillar, we discard the rest? is that the same as sampling points if we have to many?
    shuffle_index = np.random.permutation(len(point_cloud))
    point_cloud = point_cloud[shuffle_index, :]

    x_edges = np.arange(-trim_range, trim_range, pillar_size)
    y_edges = np.arange(-trim_range, trim_range, pillar_size)

    # translate point cloud such that the center of the cloud is the origin
    point_cloud = point_cloud - np.array((origin[0], origin[1], 0))

    # compute pillar for each point
    x_grids = np.floor((point_cloud[:,0] + trim_range)/pillar_size).astype(int)
    y_grids = np.floor((point_cloud[:,1] + trim_range)/pillar_size).astype(int)
    grid_pairs = list(zip(x_grids, y_grids))
    unique_grid_pairs = list(set(grid_pairs))

    # we can sample 1260 pillars here already, before computing features that may be thrown away later
    # good idea, but we still have to sort all points? so well loop through all points anyway
    '''if len(unique_grid_pairs) > max_num_pillars:
        unique_grid_pairs = np.array(list(set(grid_pairs)))
        random_index = random.sample(list(np.arange(len(unique_grid_pairs))), max_num_pillars)
        unique_grid_pairs = unique_grid_pairs[random_index]
        unique_grid_pairs = [tuple(pair) for pair in unique_grid_pairs]
        # update grids accordingly??'''

    # calculate features for all points
    # get center of each pillar for each point
    x_offset = x_edges[x_grids] + pillar_size / 2
    y_offset = y_edges[y_grids] + pillar_size / 2
    xp = point_cloud[:, 0] - x_offset
    yp = point_cloud[:, 1] - y_offset
    z = point_cloud[:, 2]
    features = np.vstack((xp.T, yp.T, z.T)).T

    pillar_feature_dict = dict.fromkeys(unique_grid_pairs)
    coordinate_dict = {}
    for row in range(len(point_cloud[:, 0])):

        key = (x_grids[row], y_grids[row])

        if pillar_feature_dict[key] is not None: # if the pillar already has at least one point
            if len(pillar_feature_dict[key]) < 100:
                cell_value = pillar_feature_dict[key]

                new_feature = features[row, :]
                cell_value = np.vstack((cell_value, new_feature))

                pillar_feature_dict.update({key: cell_value})

        else:
            new_feature = features[row, :]
            pillar_feature_dict.update({key: new_feature})

        coordinate_dict.update({key: point_cloud[row, :]})


    # CREATE TENSOR
    # Initialize feature tensor
    feature_tensor = np.zeros((dimension, max_number_of_pillars, max_number_of_points_per_pillar))
    coordinate_tensor = np.zeros((max_number_of_pillars,3))
    # 1. Check how many keys in the dict. If more than max number of pillars pick random max_numer_of_pillars
    number_of_pillars = len(pillar_feature_dict.keys())
    if number_of_pillars > max_number_of_pillars:
        key_list = random.sample(list(pillar_feature_dict), max_number_of_pillars)
    else:
        key_list = list(pillar_feature_dict)

    pillar = 0
    for key in key_list:
        # Get value from dict
        key_value = pillar_feature_dict[key]
        if len(np.shape(key_value)) == 1:
            key_value = np.expand_dims(pillar_feature_dict[key], axis=0)
        number_of_points = np.shape(key_value)[0]
        feature_tensor[:, pillar, :number_of_points] = key_value.T
        coordinate_tensor[pillar, :] = coordinate_dict[key]
        pillar += 1

    return feature_tensor, coordinate_tensor
