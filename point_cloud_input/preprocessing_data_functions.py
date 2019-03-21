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


def create_pillars(point_cloud, pillar_size=0.16):
    '''
    Function that creates a dict containing the 8 features for each pillar in a point cloud. Each pillar is grid_size large.
    :param point_cloud: <nd.array> [nx3] nd array containing the x, y, z coordinates of a point cloud
    :param pillar_size: <float> The size of the pillar.
    :return: pillar_dict: <dict> A dict containing the features for each pillar
    '''

    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:,0])
    min_y = np.min(point_cloud[:,1])
    max_y = np.max(point_cloud[:, 1])

    x_edges = []
    x = min_x

    while x <= max_x + pillar_size:  # creates list with the x-edge values of the grids
        x_edges.append(x)
        x = x + pillar_size

    y_edges = []
    y = min_y

    while y <= max_y + pillar_size:  # creates list with the y-edge values of the grids
        y_edges.append(y)
        y = y + pillar_size

    pillar_dict = {}

    for row in range(len(point_cloud[:,0])): # TODO: maybe we should sort for x and y as we did before? /A
        # Get which grid the current point belongs to. The key in the dict has the name of the grid. ex 0,0
        x_grid, y_grid = get_grid(point_cloud[row,0], point_cloud[row,1], x_edges, y_edges)
        cell_name = str(x_grid) + ',' + str(y_grid)

        # If the cell name has been used before concatenate the points and update the value of the key. Else create
        # a new key and add the coordinates of the point.
        if cell_name in pillar_dict.keys():

            cell_value = pillar_dict[cell_name]
            cell_value = np.vstack((cell_value, point_cloud[row,:]))

            pillar_dict.update({cell_name: cell_value})

        else:
            pillar_dict.update({cell_name : point_cloud[row,:]})

    # Calculate the features for each point in the point cloud.
    for key in pillar_dict.keys():

        key_value = pillar_dict[key]
        num_points = len(key_value)

        # translate all points to "new" origo in pillar
        x_grid, y_grid = get_grid(key_value[0,0], key_value[0,1], x_edges, y_edges)
        key_value[:,0] = key_value[:,0] - x_edges[x_grid]
        key_value[:,1] = key_value[:,1] - y_edges[y_grid]
        # key_value[:,2] = key_value[:,2]

        # 1. calculate distance to the arithmetic mean for x,y,z
        # And then calculate the features xc, yc, zc which is the distance from the arithmetic mean. Reshape to be able
        # to stack them later.

        # 2. calculate the offset from the pillar x,y center i.e xp and yp. TODO: I AM UNCERTAIN ABOUT THIS! //S
        # Reshape to be able to stack them later.

        if np.shape(key_value) == (3,):

            key_value = key_value.reshape((1,np.shape(key_value)[0]))

            # TODO, we could skip this and always execute the else below instead?
            '''
            x_mean = key_value[0, 0] / num_points
            y_mean = key_value[0, 1] / num_points
            z_mean = key_value[0, 2] / num_points

            xc = key_value[0, 0] - x_mean
            xc = np.array([[xc]])

            yc = key_value[0, 1] - y_mean
            yc = np.array([[yc]])

            zc = key_value[0, 2] - z_mean
            zc = np.array([[zc]])

            x_offset = pillar_size / 2 + key_value[0, 0]
            y_offset = pillar_size / 2 + key_value[0, 1]

            xp = key_value[0, 0] - x_offset
            xp = np.array([[xp]])

            yp = key_value[0, 1] - y_offset
            yp = np.array([[yp]])

            # 3. Append the new features column wise to the array with the point coordinates.
            features = np.hstack((key_value, xc, yc, zc, xp, yp))
            # 4. Update the dict key with the complete feature array
            pillar_dict.update({key: features})
            '''

       # else:
        x_mean = key_value[:, 0].sum(axis=0)/num_points #Todo: are these ALL points or just number of coordinate tripletes?
        y_mean = key_value[:, 1].sum(axis=0)/num_points
        z_mean = key_value[:, 2].sum(axis=0)/num_points

        xc = key_value[:, 0] - x_mean
        xc = xc.reshape((np.shape(xc)[0],1))

        yc = key_value[:, 1] - y_mean
        yc = yc.reshape((np.shape(yc)[0], 1))

        zc = key_value[:, 2] - z_mean
        zc = zc.reshape((np.shape(zc)[0], 1))

        #x_offset = pillar_size/2 + np.min(key_value[:, 0]) # TODO, we should be checking the edges here, not min/max values
        #y_offset = pillar_size/2 + np.min(key_value[:, 1]) # TODO, we should be checking the edges here, not min/max values

        #x_grid, y_grid = get_grid(key_value[0,0], key_value[0,1], x_edges, y_edges)
        x_offset = x_edges[x_grid] + pillar_size/2
        y_offset = y_edges[y_grid] + pillar_size/2

        xp = key_value[:, 0] - x_offset
        xp = xp.reshape((np.shape(xp)[0],1))

        yp = key_value[:, 1] - y_offset
        yp = yp.reshape((np.shape(yp)[0],1))

        # 3. Append the new features column wise to the array with the point coordinates.
        features = np.hstack((key_value, xc, yc, zc, xp, yp))
        # 4. Update the dict key with the complete feature array
        pillar_dict.update({key: features})

    return pillar_dict


def get_feature_tensor(pillar_dict, max_number_of_pillars=12000, max_number_of_points_per_pillar=100, dimension=8):
    '''
    Function that creates the feature tensor with dimension (D,P,N)
    D = Dimension (8)
    P = max antal pillars (12000)
    N = maximum points per pillar (100)
    :param pillar_dicts: <dict> Dict containing features for each pillar.
    :param max_number_of_pillars: <int> Max number of pillars in a sample. (default=12000)
    :param max_number_of_points_per_pillar: <int> Max number of points in a sample. (default=100)
    :param dimension: <int> Dimension of features. (default=8)
    :return: feature_tensor
    '''

    # Initialize feature tensor

    feature_tensor = np.zeros((dimension, max_number_of_pillars, max_number_of_points_per_pillar))

    # 1. Check how many keys in the dict. If more than max number of pillars pick random max_numer_of_pillars
    number_of_pillars = len(pillar_dict.keys())

    # in number of pillars is mor than the maximum allowed. set number of pillars = max_number and sample the key list
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

        feature_point = 0
        for point in points:
            for feature in range(0 , dimension):
                feature_tensor[feature, pillar, feature_point] = key_value[point,feature]
            feature_point += 1
        pillar += 1

    return feature_tensor

