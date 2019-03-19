import numpy as np
import pandas as pd

grid_size = 0.16


path_to_ply = '/Users/sabinalinderoth/Desktop/Ply_files_1/TEST_sorted_grid_ply_1/grid_13_10/070832.ply'
point_cloud = pd.read_csv(path_to_ply, delimiter=' ', skiprows=7, header=None, names=('x','y','z'))
point_cloud = point_cloud.values


x_min = -1.7245
x_max =  1.7245


number_x_grids = int(np.ceil((x_max - x_min) / grid_size))
number_of_x_edges = number_x_grids + 1
print('number_of_x_edges:', number_of_x_edges)
x_edges = [x_min + x for x in np.arange(number_of_x_edges) * grid_size if x < x_max + grid_size]  # creates list with all the edge values of the grids
print('length x edges', np.shape(x_edges))
print('x edges:', x_edges)

x_edges_list = []
x = x_min

while x < x_max:

    x_edges_list.append(x)

    x = x + grid_size
    print(x_edges_list)


print(np.shape(x_edges_list))
print('done')
