import numpy as np
import random
import pandas as pd
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
    :return: pillar_dict: <dict> A dict containing the features for each piller
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

    for row in range(len(point_cloud[:,0])):
        # Get which grid yhe current point belongs to. The key in the dict has the name of the grid. ex 0,0
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

        # 1. calculate distance t othe arithmetic mean for x,y,z
        # And then calculate the features xc, yc, zc which is the distance from the arithmetic mean. Reshape to be able
        # to stack them later.

        # 2. calculate the offset from the pillar x,y center i.e xp and yp. I AM UNCERTAIN ABOUT THIS! //S
        # Reshape to be able to stack them later.
        
        if np.shape(key_value) == (3,):

            key_value = key_value.reshape((1,np.shape(key_value)[0]))
            print(np.shape(key_value))

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

            print(np.shape(key_value), np.shape(key_value), np.shape(xc), np.shape(yc), np.shape(zc), np.shape(xp), np.shape(yp))
            features = np.hstack((key_value, xc, yc, zc, xp, yp))
            print(features)
            # 4. Update the dict key with the complete feature array
            pillar_dict.update({key: features})

        else:
            x_mean = key_value[:, 0].sum(axis=0)/num_points
            y_mean = key_value[:, 1].sum(axis=0)/num_points
            z_mean = key_value[:, 2].sum(axis=0)/num_points

            xc = key_value[:, 0] - x_mean
            xc = xc.reshape((np.shape(xc)[0],1))

            yc = key_value[:, 1] - y_mean
            yc = yc.reshape((np.shape(yc)[0], 1))

            zc = key_value[:, 2] - z_mean
            zc = zc.reshape((np.shape(zc)[0], 1))


            x_offset = pillar_size/2 + np.min(key_value[:, 0])
            y_offset = pillar_size/2 + np.min(key_value[:, 1])

            xp = key_value[:, 0] - x_offset
            xp = xp.reshape((np.shape(xp)[0],1))

            yp = key_value[:, 1] - y_offset
            yp = yp.reshape((np.shape(yp)[0],1))

            # 3. Append the new features column wise to the array with the point coordinates.
            features = np.hstack((key_value, xc, yc, zc, xp, yp))
            # 4. Update the dict key with the complete feature array
            pillar_dict.update({key: features})



    print('pillar dict done')
    return pillar_dict


def get_feature_tensor(pillar_dict, max_number_of_pillars=12000, max_number_of_points_per_pillar=100):
    '''
    Function that creates the feature tensor with dimension (D,P,N)
    D = Dimension (8)
    P = max antal pillars (12000)
    N = maximum points per pillar (100)
    :param pillar_dict: <dict> Dict containing features for each pillar.
    :param max_number_of_pillars: <int> Max number of pillars in a sample
    :param max_number_of_points_per_pillar: <int> Max number of points in a sample.
    :return:
    '''

    # 1. Check how many keys in the dict. If more than max number of pillars pick random max_numer_of_pillars
    number_of_pillars = len(pillar_dict.keys())

    if number_of_pillars > max_number_of_pillars:
        random_key = random.sample(pillar_dict, 10)[0]
        print(random_key)

        print('Do something')

    else:
        print('Do something')


# 2. For each pillar. Check number of points. if more than max_number_of_points_per_pillar pick random
# max_number_of_points_per_pillar.

# The tensor dimension is (D,P,N)
# D = Dimension (8)
# P = max antal pillars (12000)
# N = maximum points per pillar (100)

#    return feature_tensor


path_to_ply = '/Users/sabinalinderoth/Desktop/Ply_files_1/TEST_sorted_grid_ply_1/grid_13_10/070832.ply'
point_cloud = pd.read_csv(path_to_ply, delimiter=' ', skiprows=7, header=None, names=('x','y','z'))
point_cloud = point_cloud.values


pillar_dict = create_pillars(point_cloud)
feature_tensor = get_feature_tensor(pillar_dict, max_number_of_pillars=12000, max_number_of_points_per_pillar=100)