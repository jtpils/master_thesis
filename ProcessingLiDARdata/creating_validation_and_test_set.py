import numpy as np
import os
from lidar_data_functions import *
import pandas as pd
import csv
import shutil

path_to_ply_folder = input('Enter path to ply folder of original data:')
path_to_csv_folder = input('Enter path to csv folder of original data:')
input_folder_name = input('Type name of new folder to save the validation and test set:')
print(' ')

current_path = os.getcwd()
folder_path = os.path.join(current_path, input_folder_name)

# Path to the validation set
path_validation = os.path.join(folder_path, 'validation_set')
path_validation_ply = os.path.join(path_validation,'ply')

# Path to the test set
path_test = os.path.join(folder_path, 'test_set')
path_test_ply = os.path.join(path_test,'ply')

# Path to csv that is going to be filled
csv_labels_path_val = os.path.join(path_validation,'validation_set.csv')
csv_labels_path_test = os.path.join(path_test, 'test_set.csv')


# Create folders
try:
   os.mkdir(folder_path)  # Folder to store the validation and test map

   os.mkdir(path_validation)  # Folder to store the validation set's ply folder and csv file
   os.mkdir(path_validation_ply)  # Folder to store the validation ply files

   os.mkdir(path_test)  # Folder to store the test set's ply folder and csv file
   os.mkdir(path_test_ply)  # Folder to store the test ply files

except OSError:
    print('Failed to create new directory.')
else:
    print('Successfully created new directory with subdirectory, at ', folder_path)


# Create two new csv files one for the global coordinates of the validation set and one for the test set.
with open(csv_labels_path_val, mode='w') as csv_file:
    fieldnames = ['frame_number', 'x', 'y','z', 'yaw']
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(fieldnames)
    print('Successfully created validation set csv')


with open(csv_labels_path_test, mode='w') as csv_file:
    fieldnames = ['frame_number', 'x', 'y', 'z','yaw']
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(fieldnames)
    print('Successfully created test set csv')


global_coordinates = pd.read_csv(path_to_csv_folder)
global_coordinates = global_coordinates.values

x_max = np.max(global_coordinates[:, 1])
x_min = np.min(global_coordinates[:, 1])

y_max = np.max(global_coordinates[:, 2])
y_min = np.min(global_coordinates[:, 2])

# Fraction of town to validation set
fraction_of_town_validation = (x_max-x_min)*0.67

# sort the list after x-coordinates just to make it easer to see in the csv file that it is separated in the right way
x_sorted_global_coordinates = np.asarray(sorted(global_coordinates, key=lambda row: row[1]))

# all x_values under x_min+num_two_thirds_of_files limit should be in csv_file_val
x_limit = x_min+fraction_of_town_validation

# all values under the limit to validation set
validation_x = np.nonzero(global_coordinates[:, 1] < x_limit)
validation_set =  global_coordinates[validation_x,:]

# Fill csv with frame number, x, y, z and yaw
with open(csv_labels_path_val, mode='a') as csv_file:

    for i in range(0,np.shape(validation_set)[1]):
        frame_number = int(validation_set[0, i, 0])
        x_coord = validation_set[0, i, 1]
        y_coord = validation_set[0, i, 2]
        z_coord = validation_set[0, i, 3]
        yaw = validation_set[0, i, 4]

        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([frame_number, x_coord, y_coord, z_coord,yaw])


# Create list of all the ply files in the original data set folder
files_in_ply_folder = os.listdir(path_to_ply_folder)

# Remove the ply part
ply_files_in_original_folder = [x[:-4] for x in files_in_ply_folder]
# Convert the strings to type int
ply_files_in_original_folder_int = [int(float(x)) for x in ply_files_in_original_folder]
# Convert the values in the validation folder to int
ply_files_in_val_folder = validation_set[:, :, 0].astype(float).astype(int)

# For all the files in the validation folder find the index of the same file in the original data folder.
# Copy that ply file to the new folder.
for file in ply_files_in_val_folder[0]:

    idx = ply_files_in_original_folder_int.index(file)
    ply_file = files_in_ply_folder[idx]
    ply_file_to_move = os.path.join(path_to_ply_folder, ply_file)

    # Source folder of ply file
    source = ply_file_to_move
    # Destination folder of ply file
    destination = os.path.join(path_validation_ply, ply_file)

    shutil.copyfile(source, destination)

# all values over the limit to test set
test_x = np.nonzero(global_coordinates[:, 1] >= x_limit)
test_set = global_coordinates[test_x,:]

# Fill csv with frame number, x, y, z and yaw
with open(csv_labels_path_test, mode='a') as csv_file:

    for i in range(0,np.shape(test_set)[1]):
        frame_number = int(test_set[0, i, 0])
        x_coord = test_set[0, i, 1]
        y_coord = test_set[0, i, 2]
        z_coord = test_set[0, i, 3]
        yaw = test_set[0, i, 4]

        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([frame_number, x_coord, y_coord, z_coord, yaw])

# Convert string type to int
ply_files_in_test_folder = test_set[:, :, 0].astype(int)

# For all the files in the validation folder find the index of the same file in the original data folder.
# Copy that ply file to the new folder.
for file in ply_files_in_test_folder[0]:

    idx = ply_files_in_original_folder_int.index(file)
    ply_file = files_in_ply_folder[idx]

    ply_file_to_move = os.path.join(path_to_ply_folder, ply_file )

    # Source folder of ply file
    source = ply_file_to_move

    # Destination folder of ply file
    destination = os.path.join(path_test_ply, ply_file)
    shutil.copyfile(source, destination)
