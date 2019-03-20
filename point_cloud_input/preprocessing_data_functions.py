import numpy as np


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
        x_mean = key_value[:, 0].sum(axis=0)/num_points
        y_mean = key_value[:, 1].sum(axis=0)/num_points
        z_mean = key_value[:, 2].sum(axis=0)/num_points

        # Calculate the features xc, yc, zc which is the distance from the arithmetic mean. Reshape to be able to stack
        # them later.
        xc = key_value[:, 0] - x_mean
        xc = xc.reshape((np.shape(xc)[0],1))

        yc = key_value[:, 1] - y_mean
        yc = yc.reshape((np.shape(yc)[0], 1))

        zc = key_value[:, 2] - z_mean
        zc = zc.reshape((np.shape(zc)[0], 1))

        # 2. calculate the offset from the pillar x,y center. I AM UNCERTAIN ABOUT THIS! //S
        # Reshape to be able to stack them later.
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

    return pillar_dict
