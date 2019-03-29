from data_set_point_cloud import PointCloudDataSet
import time
import numpy as np

data_set_path = '/home/master04/Desktop/Dataset/ply_grids/Town02_sorted_grid_ply'
number_of_samples = 10
data_set = PointCloudDataSet(data_set_path, number_of_samples)

t1 = time.time()
sample = data_set.__getitem__(0)
t2 = time.time()
print('Time to load 1 sample: ', t2-t1)

sweep = sample['sweep']
map_cutout = sample['map']


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


# sort into pillars
def createPillarsFast(point_cloud, pillar_size=0.5, trim_range=15):
    t1 = time.time()
    num_points = 300  # maximum number of points in each pillar
    pillar_list = []

    #point_cloud = np.array(((1,1,1),(2,2,2),(3,3,3),(3,3,3),(1,1,1),(1,1,1)))

    xgrids = np.floor((point_cloud[:,0] + trim_range) / pillar_size)
    ygrids = np.floor((point_cloud[:,1] + trim_range) / pillar_size)
    grid_list = list(zip(xgrids, ygrids))   # return grid index for each lidar point
    unique_grids = list(set(zip(xgrids, ygrids)))  # return set of unique grid combinations. This variable should be kept around to translate the features to a pseudo image later on.

    get_indices = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

    # Sort all points into a list. One element in the list per unique grid.
    for grid in unique_grids:
        #indices = grid_list.index(grid)  # find all coordinates within this specific grid
        indices = get_indices(grid, grid_list)
        pillar_list.append(point_cloud[indices,:])

    t2 = time.time()
    print('createPillarsFast: ', t2-t1)

    return pillar_list


def create_pillars(point_cloud, pillar_size=0.5):
    t1 = time.time()
    x_edges = np.arange(-15, 15, pillar_size)
    y_edges = np.arange(-15, 15, pillar_size)
    pillar_dict = {}
    #For-loop for every coordinate tripe in the list
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

    t2 = time.time()
    print('create_pillars: ', t2-t1)
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
        x_mean = key_value[:, 0].sum(axis=0)/num_points
        y_mean = key_value[:, 1].sum(axis=0)/num_points
        z_mean = key_value[:, 2].sum(axis=0)/num_points

        xc = key_value[:, 0] - x_mean
        xc = xc.reshape((np.shape(xc)[0],1))

        yc = key_value[:, 1] - y_mean
        yc = yc.reshape((np.shape(yc)[0], 1))

        zc = key_value[:, 2] - z_mean
        zc = zc.reshape((np.shape(zc)[0], 1))

        # 2. calculate the offset from the pillar x,y center i.e xp and yp.
        # Reshape to be able to stack them later.
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



a = create_pillars(map_cutout) # 0.8 s
a = createPillarsFast(map_cutout) # 7.4 s, not so fast...
