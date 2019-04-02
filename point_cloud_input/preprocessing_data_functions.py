import numpy as np
from random import *
import random
import pandas as pd
import time
import matplotlib.pyplot as plt


def get_grid(x, y, x_edges, y_edges):
    k = 0
    for edge in x_edges:
        if x >= edge:
            x_grid_number = k
        k = k + 1

    k = 0
    for edge in y_edges:
        if y >= edge:
            y_grid_number = k
        k = k + 1

    return x_grid_number, y_grid_number


def create_pillars(point_cloud, pillar_size=0.5):
    '''
    Function that creates a dict containing the 8 features for each pillar in a point cloud. Each pillar is grid_size large.
    :param point_cloud: <nd.array> [nx3] nd array containing the x, y, z coordinates of a point cloud
    :param pillar_size: <float> The size of the pillar.
    :return: pillar_dict: <dict> A dict containing the features for each pillar
    '''

    x_edges = np.arange(-15, 15, pillar_size)
    y_edges = np.arange(-15, 15, pillar_size)

    pillar_dict = {}
    pillar_feature_dict = {}
    coordinate_dict = {}
    for row in range(len(point_cloud[:,0])): # TODO: maybe we should sort for x and y as we did before? /A
        # Get which grid the current point belongs to. The key in the dict has the name of the grid. ex 0,0
        x_grid, y_grid = get_grid(point_cloud[row,0], point_cloud[row,1], x_edges, y_edges)
        cell_name = str(x_grid) + ',' + str(y_grid)

        # If the cell name has been used before concatenate the points and update the value of the key. Else create
        # a new key and add the coordinates of the point.
        if cell_name in pillar_dict.keys():
            cell_value = pillar_dict[cell_name]
            cell_value = np.vstack((cell_value, point_cloud[row, :]))

            pillar_dict.update({cell_name: cell_value})
        else:
            pillar_dict.update({cell_name : point_cloud[row,:]})

        if cell_name not in coordinate_dict.keys():
            # NEW: save the one coordinate for each pillar
            coordinate_dict.update({cell_name : point_cloud[row,:]})

    # Calculate the features for each point in the point cloud.
    for key in pillar_dict.keys():

        key_value = pillar_dict[key]
        num_points = len(key_value)

        if np.shape(key_value) == (3,):
            key_value = key_value.reshape((1,np.shape(key_value)[0]))

        x_grid, y_grid = get_grid(key_value[0,0], key_value[0,1], x_edges, y_edges)

        # 1. calculate distance to the arithmetic mean for x,y,z
        # And then calculate the features xc, yc, zc which is the distance from the arithmetic mean. Reshape to be able
        # to stack them later.
        #x_mean = key_value[:, 0].sum(axis=0)/num_points
        #y_mean = key_value[:, 1].sum(axis=0)/num_points
        #z_mean = key_value[:, 2].sum(axis=0)/num_points

        #xc = key_value[:, 0] - x_mean
        #xc = xc.reshape((np.shape(xc)[0],1))

        #yc = key_value[:, 1] - y_mean
        #yc = yc.reshape((np.shape(yc)[0], 1))

        #zc = key_value[:, 2] - z_mean
        #zc = zc.reshape((np.shape(zc)[0], 1))

        # 1. Get the z value
        z = key_value[:, 2]
        z = z.reshape((np.shape(z)[0], 1))

        # 2. calculate the offset from the pillar x,y center i.e xp and yp.
        # Reshape to be able to stack them later.
        x_offset = x_edges[x_grid] + pillar_size/2
        y_offset = y_edges[y_grid] + pillar_size/2

        xp = key_value[:, 0] - x_offset
        xp = xp.reshape((np.shape(xp)[0],1))

        yp = key_value[:, 1] - y_offset
        yp = yp.reshape((np.shape(yp)[0],1))

        # 3. Append the new features column wise to the array with the point coordinates.
        #features = np.hstack((key_value, xc, yc, zc, xp, yp))

        # NEW: Only save the z, xp and yp
        features = np.hstack((xp, yp, z))

        # 4. Update the dict key with the complete feature array
        pillar_feature_dict.update({key: features})

    # NEW: output coordinate dict
    return pillar_feature_dict, coordinate_dict


def get_feature_tensor(pillar_dict, coordinate_dict, max_number_of_pillars=3600, max_number_of_points_per_pillar=300, dimension=3):
    '''
    Function that creates the feature tensor with dimension (D,P,N)
    D = Dimension (3) xp, yp, z
    P = max antal pillars (12000)
    N = maximum points per pillar (100)
    :param pillar_dicts: <dict> Dict containing features for each pillar.
    :param max_number_of_pillars: <int> Max number of pillars in a sample. (default=3600)
    :param max_number_of_points_per_pillar: <int> Max number of points in a sample. (default=300)
    :param dimension: <int> Dimension of features. (default=8)
    :return: feature_tensor
    '''

    # Initialize feature tensor
    feature_tensor = np.zeros((dimension, max_number_of_pillars, max_number_of_points_per_pillar))
    coordinate_tensor = np.zeros((max_number_of_pillars,3))
    # 1. Check how many keys in the dict. If more than max number of pillars pick random max_numer_of_pillars
    number_of_pillars = len(pillar_dict.keys())

    # if number of pillars is more than the maximum allowed. set number of pillars = max_number and sample the key list
    if number_of_pillars > max_number_of_pillars:

        # number_of_pillars = max_number_of_pillars
        key_list = random.sample(list(pillar_dict), max_number_of_pillars)
    else:
        key_list = list(pillar_dict)

    pillar = 0
    for key in key_list:
        # Get value from dict
        key_value = pillar_dict[key]
        number_of_points = np.shape(key_value)[0]

        if number_of_points > max_number_of_points_per_pillar:

            number_of_points_index = list(range(0,number_of_points-1))
            random_index = sample(number_of_points_index, max_number_of_points_per_pillar)

            points = [number_of_points_index[i] for i in random_index]

        else:
            points = np.array(range(0,number_of_points))

        # this can be done in one row
        lidar_point = 0
        for point in points:
            for feature in range(0 , dimension):
                feature_tensor[feature, pillar, lidar_point] = key_value[point,feature]

            lidar_point += 1
        # NEW: save the coordinates in a tensor
        coordinate_tensor[pillar,:] = coordinate_dict[key]
        pillar += 1

    # NEW: Output a tensor
    return feature_tensor, coordinate_tensor

