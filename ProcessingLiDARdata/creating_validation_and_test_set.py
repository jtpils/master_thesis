import numpy as np
import os
from lidar_data_functions import *
import pandas as pd
import csv
import shutil

# create 2 new folders. one for storing the validation set and one for storing the test set

path_to_ply_folder = '/Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/_out_Town02_190306_1/pc/' # input('Enter path to ply folder:')
input_folder_name = 'validation_test_set' # input('Type name of new folder to save data:')
print(' ')

current_path = os.getcwd()
folder_path = os.path.join(current_path, input_folder_name)

path_validation = os.path.join(folder_path, 'validation_set')
path_validation_ply = os.path.join(path_validation,'ply')

path_test = os.path.join(folder_path, 'test_set')
path_test_ply = os.path.join(path_test,'ply')

# path to csv
csv_labels_path_val = os.path.join(path_validation,'validation_set.csv')
csv_labels_path_test = os.path.join(path_test, 'test_set.csv')


# create all the folders.
try:
   os.mkdir(folder_path)  # folder to store the validation and test map

   os.mkdir(path_validation)  # Folder to store the validation plys and csv
   os.mkdir(path_validation_ply)  # Folder to store the validation ply files

   os.mkdir(path_test)  # Folder to store the test plys and csv
   os.mkdir(path_test_ply)  # Folder to store the test ply files

except OSError:
    print('Failed to create new directory.')
else:
    print('Successfully created new directory with subdirectory, at ', folder_path)

# create two new csv files one for validation and one for test

with open(csv_labels_path_val, mode='w') as csv_file:
    fieldnames = ['frame_number', 'x', 'y','z', 'yaw']
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(fieldnames)
    print('Successfully created validation set csv')


with open(csv_labels_path_test, mode='w') as csv_file:
    fieldnames = ['frame_number', 'x', 'y', 'angle']
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(fieldnames)
    print('Successfully created test set csv')



# Read csv file

global_coordinates = pd.read_csv('/Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/_out_Town02_190306_1/Town02_190306_1.csv')
global_coordinates = global_coordinates.values

x_max = np.max(global_coordinates[:, 1])
x_min = np.min(global_coordinates[:, 1])

y_max = np.max(global_coordinates[:, 2])
y_min = np.min(global_coordinates[:, 2])

# find the 67% limit
num_two_thirds_of_files = (x_max-x_min)*0.67

# sort the list after x-coordinates just to make it easer to see in the csv file that it is separated in the right way
x_sorted_global_coordinates = np.asarray(sorted(global_coordinates, key=lambda row: row[1]))

# all x_values under x_min+num_two_thirds_of_files limit should be in csv_file_val
x_limit = x_min+num_two_thirds_of_files

validation_x = np.nonzero(global_coordinates[:, 1] < x_limit)
validation_set =  global_coordinates[validation_x,:]


# write frame_number in column 1, and the transformation in the next columns
with open(csv_labels_path_val, mode='a') as csv_file:

    for i in range(0,np.shape(validation_set)[1]):
        frame_number = int(validation_set[0, i, 0])
        x_coord = validation_set[0, i, 1]
        y_coord = validation_set[0, i, 2]
        z_coord = validation_set[0, i, 3]
        yaw = validation_set[0, i, 4]

        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([frame_number, x_coord, y_coord, z_coord,yaw])


# check the frame number i csv and then find the one in the ply folder
files_in_ply_folder = os.listdir(path_to_ply_folder)

# remove the ply part
ply_files_in_original_folder = [x[:-4] for x in files_in_ply_folder]
ply_files_in_original_folder_int = [int(float(x)) for x in ply_files_in_original_folder]
ply_files_in_val_folder = validation_set[:, :, 0].astype(float).astype(int)

for file in ply_files_in_val_folder[0]:

    idx = ply_files_in_original_folder_int.index(file)

    ply_file = files_in_ply_folder[idx]

    ply_file_to_move = os.path.join(path_to_ply_folder, ply_file)

    source = ply_file_to_move
    destination = os.path.join('/Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/validation_test_set/validation_set/ply' , ply_file)

    shutil.copyfile(source, destination)

test_x = np.nonzero(global_coordinates[:, 1] >= x_limit)
test_set = global_coordinates[test_x,:]


with open(csv_labels_path_test, mode='a') as csv_file:

    for i in range(0,np.shape(test_set)[1]):
        frame_number = int(test_set[0, i, 0])
        x_coord = test_set[0, i, 1]
        y_coord = test_set[0, i, 2]
        z_coord = test_set[0, i, 3]
        yaw = test_set[0, i, 4]

        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([frame_number, x_coord, y_coord, z_coord, yaw])


ply_files_in_test_folder = test_set[:, :, 0].astype(int)

for file in ply_files_in_test_folder[0]:

    idx = ply_files_in_original_folder_int.index(file)
    ply_file = files_in_ply_folder[idx]

    ply_file_to_move = os.path.join(path_to_ply_folder, ply_file )

    source = ply_file_to_move
    destination = os.path.join('/Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/validation_test_set/test_set/ply' , ply_file)
    shutil.copyfile(source, destination)
