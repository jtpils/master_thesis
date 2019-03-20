import numpy as np
import pandas as pd

grid_size = 0.16


path_to_ply = '/Users/sabinalinderoth/Desktop/Ply_files_1/TEST_sorted_grid_ply_1/grid_13_10/070832.ply'
point_cloud = pd.read_csv(path_to_ply, delimiter=' ', skiprows=7, header=None, names=('x','y','z'))
point_cloud = point_cloud.values

pc_list = point_cloud.tolist()
print(len(pc_list[0][:]))

a = np.array([1,1,1])
print(a)

summa = a.sum(axis=0)/3
print(summa)
'''
b = np.array([2,2,2])
c = np.vstack((a,b))
d = np.array([9,9,9])

e = np.vstack((c,d))
print(e)


f = np.array([3,3,3])

f = f.reshape((3,1))
'''




#h = np.array([[5,5,5]]).T


#i = np.hstack((e,f,g,h))
#print(i)
#e_sum_0 = e.sum(axis=0)#np.sum(e, axis=0)
#print(e_sum_0)

#e_sum_1 = np.sum(e, axis=1)
#print(e_sum_1)
'''
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

while x <= x_max + grid_size:

    x_edges_list.append(x)

    x = x + grid_size
    print(x_edges_list)


print(np.shape(x_edges_list))
print('done')

'''
