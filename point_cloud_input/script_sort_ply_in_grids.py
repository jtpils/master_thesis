import os
import pandas as pd
import numpy as np
import shutil
import csv
from tqdm import tqdm


def get_file_name_from_frame_number(frame_number):
    frame_number_string = str(int(frame_number))
    number_of_char = len(frame_number_string)
    add_number_of_zeros = 6 - number_of_char
    file_name = '0'*add_number_of_zeros + frame_number_string + '.ply'

    return file_name

#**********# change the rows marked with this #**********#

# path_to_ply = '/Users/annikal/Documents/master_thesis/ProcessingLiDARdata/_out_Town02_190221_1'#'/home/master04/Desktop/Ply_files/_out_Town03_190306_1'#**********#
# path_to_csv = os.path.join(path_to_ply, 'Town02_190221_1.csv')#**********#
path_to_pc = os.path.join(path_to_ply, 'pc')

# new_folder = '/Users/annikal/Desktop/Ply_files/TEST_sorted_grid_ply' #'/home/master04/Desktop/Ply_files/Town03_sorted_grid_ply'#**********#
os.mkdir(new_folder)

global_coordinates = pd.read_csv(path_to_csv)
min_x = np.min(global_coordinates['x'].values)
max_x = np.max(global_coordinates['x'].values)
min_y = np.min(global_coordinates['y'].values)
max_y = np.max(global_coordinates['y'].values)

grid_size = 15
number_x_grids = int(np.ceil((max_x - min_x)/grid_size))
number_of_x_edges = number_x_grids + 1
x_edges = [min_x + x for x in np.arange(number_of_x_edges) * grid_size if x < max_x + grid_size] #creates list with all the edge values of the grids
number_y_grids = int(np.ceil((max_y - min_y)/grid_size))
number_of_y_edges = number_y_grids + 1
y_edges = [min_y + y for y in np.arange(number_of_y_edges) * grid_size if y < max_y + grid_size] #creates list with all the edge values of the grids

csv_edges_path = os.path.join(new_folder, 'edges.npy')
np.save(csv_edges_path, [x_edges, y_edges])

# create all subdirectories to store ply-files in
for x in np.arange(number_x_grids):
    for y in np.arange(number_y_grids):
        grid_name = new_folder + '/grid_' + str(x) + '_' + str(y)
        os.mkdir(grid_name)




# create csv, where we store all frame numbers and global coordinates
new_csv_global_coordinates_path = os.path.join(new_folder, 'global_coordinates.csv')
with open(new_csv_global_coordinates_path, mode='w') as csv_file:
    fieldnames = ['frame_number', 'x', 'y', 'z', 'yaw']
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(fieldnames)
    print('Successfully created global_coordinates.csv')




def get_grid(x, y):
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





# loop trough each ply file and copy it to the correct directory
for row in tqdm(global_coordinates.values):
    file_name = get_file_name_from_frame_number(row[0])

    try:  # because we are out of bounds sometimes??
        x_grid, y_grid = get_grid(row[1], row[2])
        grid_name = new_folder + '/grid_' + str(x_grid) + '_' + str(y_grid)
        source = os.path.join(path_to_pc, file_name)
        destination = os.path.join(grid_name, file_name)

        try:  # beacuse some files seem to be missing
            shutil.copyfile(source, destination)

            # write coordinates from the old csv_file (already available in variable row)
            with open(new_csv_global_coordinates_path, mode='a') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([int(row[0]), row[1], row[2], row[3], row[4]])

        except:
            continue

    except:
        continue
