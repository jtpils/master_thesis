import numpy as np
import pandas as pd

def get_grid(x, y, x_edges, y_edges):
    k = 0
    for edge in x_edges:
        if x > edge:
            x_grid_number = k
        k = k + 1

    k = 0
    for edge in y_edges:
        if y > edge:
            y_grid_number = k
        k = k + 1

    return x_grid_number, y_grid_number

def create_pillars(point_cloud, number_of_points_per_pillar, number_of_non_empty_pillars, num_features=8, grid_size=0.16):
    '''
    :param point_cloud:
    :param number_of_points_per_pillar:
    :param number_of_non_empty_pillars:
    :param num_features:  <int>. Number of input features x, y, z, xc, yc, zc, xp, yp
    :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
    :return:
    '''

    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:,0])
    min_y = np.min(point_cloud[:,1])
    max_y = np.max(point_cloud[:, 1])
    min_z = np.min(point_cloud[:,2])
    max_z = np.max(point_cloud[:,2])

    print('min x:', min_x, 'max_x:', max_x, 'min y:', min_y, 'max_y:',max_y)
    pc_range = [min_x, max_x, min_y, max_y, min_z, max_z]

    number_of_cells = int(np.ceil((max_x - min_x) / grid_size))

    number_x_grids = int(np.ceil((max_x - min_x) / grid_size))
    number_of_x_edges = number_x_grids + 1
    print('number_of_x_edges:', number_of_x_edges)
    x_edges = [min_x + x for x in np.arange(number_of_x_edges) * grid_size if x < max_x + grid_size]  # creates list with all the edge values of the grids
    print('first x edges:', x_edges[:10])
    print('last x edges:', x_edges[-5:])
    number_y_grids = int(np.ceil((max_y - min_y) / grid_size))
    number_of_y_edges = number_y_grids + 1
    print('number_of_y_edges:', number_of_y_edges)
    y_edges = [min_y + y for y in np.arange(number_of_y_edges) * grid_size if y < max_y + grid_size]  # creates list with all the edge values of the grids
    print('first y edges:', y_edges[:10])
    print('last y edges:', y_edges[-5:])

    # create a dict with number_of_cells*number_of_cells cells
    # Get cell where to put the point and save it in a dict. The name of the cell should be the index.
    pillar_dict = {}
    for row in range(len(point_cloud[:,0])):

        print('point:',point_cloud[row,0], point_cloud[row,1])
        x_grid, y_grid = get_grid(point_cloud[row,0], point_cloud[row,1], x_edges, y_edges)
        print('grid:' ,x_grid, y_grid)
        cell_name = str(x_grid) + ',' + str(y_grid)
        pillar_dict.update({cell_name : point_cloud[row,:]})




        # sort x value and check y where to put it.

    print(pillar_dict)




    #pillar_tensor = np.zeros((num_features,number_of_cells,number_of_cells,number_of_points_per_pillar,8))

    # Channel 0,1,2: x,y,z coordninates
    # Channel 3,4,5:distance to arithmetic mean x,y,z respectively
    # Channel 3,4: offset from pillar, x,y center






    #print(np.shape(pillar_tensor))



number_of_points_per_pillar = 100
number_of_non_empty_pillars = 3

path_to_ply = '/Users/sabinalinderoth/Desktop/Ply_files/TEST_sorted_grid_ply/grid_13_10/070832.ply'
point_cloud = pd.read_csv(path_to_ply, delimiter=' ', skiprows=7, header=None, names=('x','y','z'))
point_cloud = point_cloud.values


t = create_pillars(point_cloud, number_of_points_per_pillar,number_of_non_empty_pillars)






